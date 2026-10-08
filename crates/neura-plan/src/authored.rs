use crate::access::Reads;
use crate::lower::Task;
use crate::span::{Measure, Split};
use neura_abi::{Kind, MeasureFields, MeasureRecord, NO_VALUE, PatchFields, PatchRecord, measure};
use neura_graph::ValueInfo;
use neura_profile::MatmulTile;
use std::collections::BTreeMap;

pub(crate) struct Authored {
    slots: Vec<u32>,
    value_slots: Vec<Vec<u32>>,
    measure_slots: Vec<Vec<u32>>,
    measures: Vec<MeasureRecord>,
}

impl Authored {
    pub(crate) fn carries(&self) -> bool {
        self.slots.iter().any(|count| *count != NO_VALUE)
    }

    pub(crate) fn slots(&self) -> &[u32] {
        &self.slots
    }

    pub(crate) fn values_of(&self, value: u32) -> &[u32] {
        &self.value_slots[value as usize]
    }

    pub(crate) fn measures(&self) -> &[MeasureRecord] {
        &self.measures
    }

    fn walks(&self, slot: u32) -> bool {
        self.slots.get(slot as usize).copied().unwrap_or(NO_VALUE) != NO_VALUE
    }

    fn count_of(&self, slot: u32) -> u32 {
        self.slots[slot as usize]
    }
}

pub(crate) fn analyse(
    slots: &[u32],
    values: &[ValueInfo],
    tasks: &mut [Task],
    measures: &[Measure],
    tiles: &[MatmulTile],
) -> Authored {
    let mut authored = Authored {
        slots: slots.to_vec(),
        value_slots: vec![Vec::new(); values.len()],
        measure_slots: vec![Vec::new(); measures.len()],
        measures: Vec::new(),
    };
    if !authored.carries() {
        authored.measures = measure_records(measures, tiles);
        return authored;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (index, measure) in measures.iter().enumerate() {
            let mut slots = Vec::new();
            match *measure {
                Measure::Elements(value)
                | Measure::Rows(value)
                | Measure::Words(value)
                | Measure::Tokens(value)
                | Measure::Tiles { value, .. } => {
                    slots.extend_from_slice(&authored.value_slots[value as usize])
                }
            }
            changed |= merge(&mut authored.measure_slots[index], slots);
        }
        for (id, info) in values.iter().enumerate() {
            let mut slots = Vec::new();
            for axis in 0..neura_abi::MAX_RANK {
                if let Some(slot) = info.shape.free(axis)
                    && authored.walks(slot)
                {
                    slots.push(slot);
                }
            }
            if info.strides_source.is_some() {
                slots.extend_from_slice(&authored.value_slots[info.storage as usize]);
            }
            changed |= merge(&mut authored.value_slots[id], slots);
        }
    }
    for task in tasks.iter_mut() {
        let mut slots = touched(task)
            .iter()
            .flat_map(|value| authored.value_slots[*value as usize].iter().copied())
            .collect::<Vec<u32>>();
        if let Some(measure) = split_measure(task.split) {
            slots.extend_from_slice(&authored.measure_slots[measure as usize]);
        }
        slots.sort_unstable();
        slots.dedup();
        task.depends = slots.iter().map(|slot| authored.count_of(*slot)).collect();
        task.depends.sort_unstable();
        task.depends.dedup();
    }
    for task in tasks.iter() {
        if task.kind != Kind::Extend {
            continue;
        }
        let walked = task.inputs[0];
        assert!(
            values[walked as usize].shape.dynamic(),
            "a {} task lands the gradient of value {} in the layout of the storage that owns it, and value {walked} walks no free extent a binding rules: the device stops at the dims it reads, and only a walk that moves with a binding tells the task where the numbers it holds end",
            task.kind.name(),
            task.inputs[1],
        );
    }
    authored.measures = measure_records(measures, tiles);
    authored
}

pub(crate) struct Counted {
    pub(crate) count: u32,
    pub(crate) writer: u32,
    pub(crate) slots: Vec<u32>,
    pub(crate) segment: u32,
    pub(crate) values: Vec<u32>,
    pub(crate) tasks: Vec<u32>,
}

pub(crate) struct Patches {
    pub(crate) owners: Vec<u32>,
    pub(crate) counted: Vec<Counted>,
}

pub(crate) struct Tables {
    pub(crate) patches: Vec<PatchRecord>,
    pub(crate) list: Vec<u32>,
}

pub(crate) fn patches(authored: &Authored, values: &[ValueInfo], tasks: &[Task]) -> Patches {
    let mut owners = vec![NO_VALUE; tasks.len()];
    let mut counted = Vec::new();
    if !authored.carries() {
        return Patches { owners, counted };
    }
    let mut grouped = BTreeMap::<u32, Vec<u32>>::new();
    for slot in 0..authored.slots.len() as u32 {
        if authored.walks(slot) {
            grouped
                .entry(authored.count_of(slot))
                .or_default()
                .push(slot);
        }
    }
    for (count, slots) in grouped {
        let writers = tasks
            .iter()
            .enumerate()
            .filter(|(_, task)| writes(task, count))
            .map(|(index, _)| index)
            .collect::<Vec<usize>>();
        assert_eq!(
            writers.len(),
            1,
            "the device counts value {count} of {} extents, and {} tasks of the plan write it; the task that counts an extent survives every fold",
            slots.len(),
            writers.len(),
        );
        let writer = writers[0];
        assert!(
            !tasks[writer].depends.contains(&count),
            "the device counts value {count} of {} extents, and the task that writes it walks a length it authors",
            slots.len(),
        );
        let closes = tasks[writer].kind == Kind::PrefixClose
            && values[tasks[writer].inputs[0] as usize].shape.elements()
                > values[tasks[writer].inputs[1] as usize].shape.elements();
        let segment = if closes {
            tasks[writer].inputs[0]
        } else {
            NO_VALUE
        };
        let values_ruled = authored
            .value_slots
            .iter()
            .enumerate()
            .filter(|(_, ruled)| ruled.iter().any(|slot| slots.contains(slot)))
            .map(|(id, _)| id as u32)
            .collect::<Vec<u32>>();
        let tasks_ruled = tasks
            .iter()
            .enumerate()
            .filter(|(_, task)| {
                let walks_the_measure = split_measure(task.split).is_some_and(|measure| {
                    authored.measure_slots[measure as usize]
                        .iter()
                        .any(|slot| slots.contains(slot))
                });
                let walks_a_segment = (task.grid != NO_VALUE || task.segments != NO_VALUE)
                    && task.depends.contains(&count);
                walks_the_measure || walks_a_segment
            })
            .map(|(index, _)| index as u32)
            .collect::<Vec<u32>>();
        if segment != NO_VALUE {
            for (index, task) in tasks.iter().enumerate() {
                if task.grid != segment && task.segments != segment {
                    continue;
                }
                assert!(
                    tasks_ruled.contains(&(index as u32)),
                    "a {} task walks the segments value {} closes, and the patch that closes that axis hands it no rows; every task that walks a segment stands on the count that rules it",
                    task.kind.name(),
                    task.segments,
                );
            }
        }
        assert_eq!(
            owners[writer], NO_VALUE,
            "one task writes the counts of two device extents, and a task carries one patch",
        );
        owners[writer] = counted.len() as u32;
        counted.push(Counted {
            count,
            writer: writer as u32,
            slots,
            segment,
            values: values_ruled,
            tasks: tasks_ruled,
        });
    }
    Patches { owners, counted }
}

pub(crate) fn tables(patches: &Patches, order: &[u32]) -> Tables {
    let mut seat = vec![NO_VALUE; patches.owners.len()];
    for (position, task) in order.iter().enumerate() {
        seat[*task as usize] = position as u32;
    }
    let mut list = Vec::<u32>::new();
    let mut records = Vec::<PatchRecord>::new();
    for counted in &patches.counted {
        assert_ne!(
            seat[counted.writer as usize], NO_VALUE,
            "the task that authors the count of value {} stands in no segment of the plan",
            counted.count,
        );
        let slots_first = list.len() as u32;
        list.extend_from_slice(&counted.slots);
        let slots_count = list.len() as u32 - slots_first;
        let values_first = list.len() as u32;
        list.extend_from_slice(&counted.values);
        let values_count = list.len() as u32 - values_first;
        let tasks_first = list.len() as u32;
        for index in &counted.tasks {
            let seat = seat[*index as usize];
            if seat == NO_VALUE {
                continue;
            }
            list.push(seat);
        }
        let tasks_count = list.len() as u32 - tasks_first;
        let record = PatchRecord::of(PatchFields {
            slots: slots_first,
            slots_count,
            count: counted.count,
            segment: counted.segment,
            values: values_first,
            values_count,
            tasks: tasks_first,
            tasks_count,
        });
        records.push(record);
    }
    Tables {
        patches: records,
        list,
    }
}

fn writes(task: &Task, value: u32) -> bool {
    task.writes().any(|written| written == value)
}

fn measure_records(measures: &[Measure], tiles: &[MatmulTile]) -> Vec<MeasureRecord> {
    measures
        .iter()
        .map(|measure| match *measure {
            Measure::Elements(value) => record(measure::ELEMENTS, value, 0, 0),
            Measure::Rows(value) => record(measure::ROWS, value, 0, 0),
            Measure::Words(value) => record(measure::WORDS, value, 0, 0),
            Measure::Tokens(value) => record(measure::TOKENS, value, 0, 0),
            Measure::Tiles { value, geometry } => {
                let tile = tiles[geometry as usize];
                record(measure::TILES, value, tile.rows(), tile.columns())
            }
        })
        .collect()
}

fn record(kind: u32, value: u32, rows: u32, columns: u32) -> MeasureRecord {
    MeasureRecord::of(MeasureFields {
        kind,
        value,
        rows,
        columns,
    })
}

fn merge(kept: &mut Vec<u32>, slots: Vec<u32>) -> bool {
    let before = kept.len();
    kept.extend(slots);
    kept.sort_unstable();
    kept.dedup();
    kept.len() != before
}

fn touched(task: &Task) -> Vec<u32> {
    let mut values = task
        .reads()
        .chain(task.writes())
        .filter(|value| *value != NO_VALUE)
        .collect::<Vec<u32>>();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn split_measure(split: Split) -> Option<u32> {
    match split {
        Split::Range { .. } => None,
        Split::Ragged { .. } => None,
        Split::Uniform { measure, .. }
        | Split::Plane { measure, .. }
        | Split::Segment { measure, .. } => Some(measure),
    }
}
