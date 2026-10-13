use crate::host::{HOST_SLICE, HostAdvance};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, ThreadId};

pub struct QueueLock {
    gate: Arc<Gate>,
}

pub(crate) struct Gate {
    held: Mutex<Option<(ThreadId, usize)>>,
    ready: Condvar,
    advance: Mutex<Option<HostAdvance>>,
}

impl Gate {
    pub(crate) fn of(advance: Option<HostAdvance>) -> Arc<Self> {
        Arc::new(Self {
            held: Mutex::new(None),
            ready: Condvar::new(),
            advance: Mutex::new(advance),
        })
    }

    pub(crate) fn advancing(&self, advance: Option<HostAdvance>) {
        *self.advance.lock().unwrap_or_else(PoisonError::into_inner) = advance;
    }

    pub(crate) fn advance(&self) -> Option<HostAdvance> {
        self.advance
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub(crate) fn take(self: &Arc<Self>) -> QueueLock {
        let thread = thread::current().id();
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((owner, depth)) = *held
            && owner == thread
        {
            *held = Some((owner, depth + 1));
            return QueueLock {
                gate: Arc::clone(self),
            };
        }
        while held.is_some() {
            held = match self.advance() {
                None => self
                    .ready
                    .wait(held)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(advance) => {
                    let (released, _) = self
                        .ready
                        .wait_timeout(held, HOST_SLICE)
                        .unwrap_or_else(PoisonError::into_inner);
                    drop(released);
                    advance.advance();
                    self.held.lock().unwrap_or_else(PoisonError::into_inner)
                }
            };
        }
        *held = Some((thread, 1));
        QueueLock {
            gate: Arc::clone(self),
        }
    }
}

impl Drop for QueueLock {
    fn drop(&mut self) {
        let mut held = self
            .gate
            .held
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match *held {
            Some((owner, depth)) if depth > 1 => *held = Some((owner, depth - 1)),
            _ => {
                *held = None;
                self.gate.ready.notify_one();
            }
        }
    }
}
