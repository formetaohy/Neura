use super::{Program, Run, Runtime};
use neura_gpu::Submission;
use neura_plan::{Encoding, Plan};
use std::sync::Arc;

pub struct Cursor {
    program: u64,
    plan: Arc<Plan>,
    encoding: Arc<Encoding>,
    wave: u32,
}

impl Cursor {
    pub fn wave(&self) -> u32 {
        self.wave
    }
}

pub struct Session<'a> {
    runtime: &'a Runtime,
    program: &'a mut Program,
    encoding: Arc<Encoding>,
    wave: u32,
    held: Option<Run>,
}

impl Runtime {
    pub fn resume<'a>(&'a self, program: &'a mut Program, cursor: Cursor) -> Session<'a> {
        assert_eq!(
            cursor.program,
            program.instance(),
            "a session resumes the very program it paused",
        );
        assert!(
            Arc::ptr_eq(&cursor.plan, &program.plan()),
            "a session resumes the program it paused, and this cursor walks another plan",
        );
        assert!(
            Arc::ptr_eq(&cursor.encoding, &program.encoding()),
            "a session resumes the binding it paused, and this program was bound again",
        );
        assert!(
            cursor.wave <= cursor.encoding.wave_count(),
            "a session resumes at wave {} of a plan that walks {} waves",
            cursor.wave,
            cursor.encoding.wave_count(),
        );
        self.own(program, Some(cursor.encoding), cursor.wave)
    }

    pub fn session<'a>(&'a self, program: &'a mut Program) -> Session<'a> {
        self.own(program, None, 0)
    }

    fn own<'a>(
        &'a self,
        program: &'a mut Program,
        encoding: Option<Arc<Encoding>>,
        wave: u32,
    ) -> Session<'a> {
        self.assert_owns(program);
        program.assert_current();
        assert!(
            program.is_compiled(),
            "a session walks a plan a compile has compiled",
        );
        assert!(
            program.windows().is_empty(),
            "a session walks a plan whose weights stay resident, and this plan streams {} weight windows",
            program.windows().len(),
        );
        assert!(
            program.wave_count() > 0,
            "a session walks a plan of at least one wave, and this plan holds none",
        );
        let encoding = encoding.unwrap_or_else(|| program.encoding());
        assert_eq!(
            encoding.wave_count(),
            program.wave_count(),
            "a session walks the waves the program holds, and this program walks {} where its binding walks {}",
            program.wave_count(),
            encoding.wave_count(),
        );
        Session {
            runtime: self,
            program,
            encoding,
            wave,
            held: None,
        }
    }
}

impl Session<'_> {
    pub fn wave(&self) -> u32 {
        self.wave
    }

    pub fn pause(mut self) -> Cursor {
        self.settle();
        Cursor {
            program: self.program.instance(),
            plan: self.program.plan(),
            encoding: self.encoding.clone(),
            wave: self.wave,
        }
    }

    pub fn remaining(&self) -> u32 {
        self.encoding.wave_count() - self.wave
    }

    pub fn done(&self) -> bool {
        self.wave >= self.encoding.wave_count()
    }

    pub fn submit(&mut self, waves: u32) -> bool {
        assert!(waves > 0, "a session submits at least one wave");
        self.settle();
        self.program.assert_current();
        self.runtime.context.assert_alive();
        assert!(
            Arc::ptr_eq(&self.encoding, &self.program.encoding()),
            "a session walks the binding it started with, and this program was bound again inside the session",
        );
        let encoding = self.encoding.clone();
        let total = encoding.wave_count();
        let sizes = encoding.wave_segments();
        let table = encoding.wave_table();
        let waits_for_the_wave_before = |wave: u32| table[1 + wave as usize * 4 + 2] != 0;
        let mut start = self.wave;
        while start < total && sizes[start as usize] == 0 {
            start += 1;
        }
        if start >= total {
            self.wave = total;
            return false;
        }
        let end = start.saturating_add(waves).min(total);
        let first = sizes.iter().take(start as usize).sum::<u32>();
        let last = first + sizes[start as usize..end as usize].iter().sum::<u32>();
        assert!(
            last > first,
            "a session walks waves {start} to {end} of a plan that holds no task there",
        );
        let segments = encoding.segments();
        let mut roots = Vec::new();
        let mut counts = Vec::with_capacity((last - first) as usize);
        let mut waves = vec![0u32; total as usize];
        for (offset, held) in segments[first as usize..last as usize].iter().enumerate() {
            let segment = first as usize + offset;
            assert!(
                held.wave >= start && held.wave < end,
                "a session walks segment {segment} of wave {} outside the waves {start} to {end} it submits",
                held.wave,
            );
            let count = encoding.predecessors(segment, first as usize);
            if count == 0 && (held.wave == start || !waits_for_the_wave_before(held.wave)) {
                roots.push(segment as u32);
            }
            counts.push(count);
            waves[held.wave as usize] += 1;
        }
        assert!(
            waves.iter().take(start as usize).all(|held| *held == 0)
                && waves.iter().skip(end as usize).all(|held| *held == 0),
            "a session submits the tasks of the waves it walks alone",
        );
        let queue = self.runtime.context.queue();
        self.program.write_extents(queue);
        if self.program.dynamic() && self.program.records_pending() {
            self.program.write_records(queue);
            self.program.records_written();
        }
        let (generation, clear) = self.program.begin();
        self.program.write_dispatch(
            queue,
            super::Dispatch {
                generation,
                clear,
                first,
                count: last - first,
                counts: Some(&counts),
                roots: &roots,
                waves: &waves,
                wave_table: encoding.successors().len() as u32,
                wave: start,
            },
        );
        let group = self.program.group();
        let mut submission = Submission::new(self.runtime.context.device(), "neura session");
        submission.dispatch(
            &self.program.resident.kernel,
            &group,
            [self.program.workgroups(), 1, 1],
        );
        let submission = submission.submit(queue);
        self.program.used(submission);
        self.held = Some(Run {
            queue: queue.clone(),
            submissions: vec![submission],
        });
        self.wave = end;
        true
    }

    pub fn wait(&mut self) -> Option<f64> {
        self.held.take().map(Run::seconds)
    }

    fn settle(&mut self) {
        if let Some(held) = self.held.take() {
            held.seconds();
        }
    }
}
