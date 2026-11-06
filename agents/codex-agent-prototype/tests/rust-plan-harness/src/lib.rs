//! Compile the real generic prefix helper with small protocol type stubs.
//! This does not replace full core/app-server integration checks.
extern crate self as codex_protocol;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ResponseItemId(String);
impl ResponseItemId {
    pub fn with_suffix(prefix: &str, id: uuid::Uuid) -> Self {
        Self(format!("{prefix}_{id}"))
    }
}

pub mod models {
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub enum ContentItem {
        InputText { text: String },
        OutputText { text: String, annotations: Option<Vec<serde_json::Value>>, logprobs: Option<Vec<serde_json::Value>> },
        Other,
    }
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub enum ResponseItem {
        Message {
            id: Option<crate::ResponseItemId>,
            role: String,
            content: Vec<ContentItem>,
            encrypted_content: Option<String>,
            status: Option<String>,
            phase: Option<String>,
            internal_chat_message_metadata_passthrough: Option<String>,
        },
        Other,
    }
}

#[path = "../../../codex/codex-rs/core/src/request_prefix.rs"]
mod request_prefix;
