use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Member {
    pub fingerprint: SmartsFingerprint,
    pub metric: Option<f32>,
}

impl Member {
    pub fn new(fingerprint: SmartsFingerprint) -> Member {
        Member {
            fingerprint,
            metric: None,
        }
    }
}

impl Display for Member {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fingerprint_string = self.fingerprint.to_string();

        write!(f, "{}", fingerprint_string)
    }
}
