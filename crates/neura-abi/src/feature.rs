use bitflags::bitflags;
use std::fmt::{self, Display, Formatter};

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Features: u32 {
        const SUBGROUP = 1;
    }
}

impl Features {
    fn names(self) -> Vec<&'static str> {
        Self::LABELS
            .iter()
            .filter(|(feature, _)| self.contains(*feature))
            .map(|(_, name)| *name)
            .collect()
    }

    const LABELS: &'static [(Self, &'static str)] = &[(Self::SUBGROUP, "subgroup")];
}

impl Display for Features {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        let names = self.names();
        if names.is_empty() {
            return out.write_str("no feature");
        }
        out.write_str(&names.join(", "))
    }
}

impl std::fmt::Debug for Features {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        let names = self.names();
        if names.is_empty() {
            return out.write_str("Features::empty()");
        }
        write!(out, "Features({})", names.join(" | "))
    }
}
