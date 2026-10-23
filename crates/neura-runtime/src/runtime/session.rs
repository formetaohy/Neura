use super::{Program, Run, Runtime, SegmentSlice};
use crate::program::WeightSchedule;
use neura_plan::{Encoding, Plan};
use std::sync::Arc;

fn wave_at(encoding: &Encoding, position: u32) -> u32 {
    encoding
        .segments()
        .get(position as usize)
        .map_or_else(|| encoding.wave_count(), |segment| segment.wave)
}

fn segment_of_wave(encoding: &Encoding, wave: u32) -> u32 {
    encoding
        .segments()
        .partition_point(|segment| segment.wave < wave) as u32
}

pub struct Cursor {
    program: u64,
    plan: Arc<Plan>,
    encoding: Arc<Encoding>,
    position: u32,
    held: Option<Run>,
}

impl Cursor {
    pub fn wave(&self) -> u32 {
        wave_at(&self.encoding, self.position)
    }

    pub fn done(&self) -> bool {
        self.position as usize >= self.encoding.segments().len()
    }

    pub fn ready(&self) -> bool {
        self.held.as_ref().is_none_or(Run::complete)
    }

    pub fn settle(&mut self) -> Option<f64> {
        self.held.take().map(Run::seconds)
    }

    pub fn wait(mut self) -> Option<f64> {
        self.settle()
    }
}

pub struct Session<'a> {
    runtime: &'a Runtime,
    program: &'a mut Program,
    encoding: Arc<Encoding>,
    schedule: Option<Arc<WeightSchedule>>,
    position: u32,
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
            cursor.ready(),
            "a session resumes the slice it paused after that slice completes; poll Cursor::ready before resuming",
        );
        assert!(
            cursor.position as usize <= cursor.encoding.segments().len(),
            "a session resumes at segment {} of a plan that walks {}",
            cursor.position,
            cursor.encoding.segments().len(),
        );
        self.own(program, Some(cursor.encoding), cursor.position)
    }

    pub fn session<'a>(&'a self, program: &'a mut Program) -> Session<'a> {
        self.own(program, None, 0)
    }

    fn own<'a>(
        &'a self,
        program: &'a mut Program,
        encoding: Option<Arc<Encoding>>,
        position: u32,
    ) -> Session<'a> {
        self.assert_owns(program);
        program.assert_current();
        assert!(
            program.is_compiled(),
            "a session walks a plan a compile has compiled",
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
        assert!(
            position as usize <= encoding.segments().len(),
            "a session starts at segment {position} of a plan that walks {}",
            encoding.segments().len(),
        );
        let schedule = program.streams_weights().then(|| program.windows());
        if let Some(schedule) = &schedule {
            program.store().pin(schedule.pins());
        }
        Session {
            runtime: self,
            program,
            encoding,
            schedule,
            position,
            held: None,
        }
    }
}

impl Session<'_> {
    pub fn wave(&self) -> u32 {
        wave_at(&self.encoding, self.position)
    }

    pub fn pause(mut self) -> Cursor {
        Cursor {
            program: self.program.instance(),
            plan: self.program.plan(),
            encoding: self.encoding.clone(),
            position: self.position,
            held: self.held.take(),
        }
    }

    pub fn remaining(&self) -> u32 {
        self.encoding.wave_count() - self.wave()
    }

    pub fn done(&self) -> bool {
        self.position as usize >= self.encoding.segments().len()
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
        let segments = self.encoding.segments();
        if self.position as usize >= segments.len() {
            return false;
        }
        let start_wave = segments[self.position as usize].wave;
        let wanted_wave = start_wave
            .saturating_add(waves)
            .min(self.encoding.wave_count());
        let wanted = segment_of_wave(&self.encoding, wanted_wave);
        let (end, pages, writes) = match &self.schedule {
            Some(schedule) => {
                assert!(
                    Arc::ptr_eq(schedule, &self.program.windows()),
                    "a session walks the weight windows it started with, and this program declared other gather rows inside the session",
                );
                let window = schedule.group_of(self.position);
                let rows = self.program.row_pages();
                let allowance = self.program.row_page_allowance();
                assert!(
                    rows.len() <= allowance as usize,
                    "a host declares {} weight pages for the rows it gathers, and the window schedule reserves {allowance}; raise the resident weight budget",
                    rows.len(),
                );
                (
                    wanted.min(window.end_segment()),
                    Runtime::merge_pages(window.pages(), &rows),
                    window.writes().to_vec(),
                )
            }
            None => (wanted, Vec::new(), Vec::new()),
        };
        assert!(
            end > self.position,
            "a session submits segments {} to {end} of a plan that holds no task there",
            self.position,
        );
        let queue = self.runtime.context.queue();
        self.program.write_extents(queue);
        if self.program.dynamic() && self.program.records_pending() {
            self.program.write_records(queue);
            self.program.records_written();
        }
        let submissions = self.runtime.dispatch(
            &*self.program,
            SegmentSlice {
                first: self.position,
                end,
                wave: start_wave,
                pages: &pages,
                writes: &writes,
                precomputed: None,
            },
        );
        self.position = end;
        self.held = Some(Run {
            queue: queue.clone(),
            submissions,
        });
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
