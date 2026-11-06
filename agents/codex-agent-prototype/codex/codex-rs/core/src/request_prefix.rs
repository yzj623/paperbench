//! Controller-owned, request-only textual messages. No files or experiment semantics.
use codex_protocol::ResponseItemId;
use codex_protocol::models::{ContentItem, ResponseItem};
use uuid::Uuid;

#[derive(Default)]
pub(crate) struct RequestPrefix {
    messages: tokio::sync::RwLock<Vec<ResponseItem>>,
}

impl RequestPrefix {
    pub(crate) async fn replace(&self, mut messages: Vec<ResponseItem>) -> Result<(), String> {
        validate(&mut messages)?;
        *self.messages.write().await = messages;
        Ok(())
    }

    pub(crate) async fn snapshot(&self) -> Vec<ResponseItem> {
        self.messages.read().await.clone()
    }
}

/// Validate the complete replacement before mutating thread state.
pub(crate) fn validate(items: &mut [ResponseItem]) -> Result<(), String> {
    for (index, item) in items.iter_mut().enumerate() {
        let ResponseItem::Message {
            id,
            role,
            content,
            encrypted_content,
            status,
            phase,
            internal_chat_message_metadata_passthrough,
        } = item
        else {
            return Err(format!(
                "messages[{index}] must be a text message, not a tool or other item"
            ));
        };
        if !matches!(role.as_str(), "user" | "developer" | "assistant") {
            return Err(format!(
                "messages[{index}].role must be user, developer or assistant; base instructions are separate"
            ));
        }
        if content.is_empty()
            || content.iter().any(|part| match part {
                ContentItem::InputText { text } => text.trim().is_empty(),
                ContentItem::OutputText {
                    text,
                    annotations,
                    logprobs,
                } => text.trim().is_empty() || annotations.is_some() || logprobs.is_some(),
                _ => true,
            })
        {
            return Err(format!(
                "messages[{index}].content must contain non-empty text only"
            ));
        }
        if encrypted_content.is_some()
            || status.is_some()
            || phase.is_some()
            || internal_chat_message_metadata_passthrough.is_some()
        {
            return Err(format!(
                "messages[{index}] must not contain response metadata"
            ));
        }
        let serialized = serde_json::to_vec(&(index, role, content)).map_err(|err| err.to_string())?;
        *id = Some(ResponseItemId::with_suffix(
            "msg",
            Uuid::new_v5(&Uuid::NAMESPACE_OID, &serialized),
        ));
    }
    Ok(())
}

pub(crate) fn prepend(input: &mut Vec<ResponseItem>, prefix: &[ResponseItem]) {
    input.splice(0..0, prefix.iter().cloned());
}

#[cfg(test)]
mod tests {
    use super::*;
    fn message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".into(),
            content: vec![ContentItem::InputText { text: text.into() }],
            encrypted_content: None,
            status: None,
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }
    #[test]
    fn ordered_request_copy_preserves_history_and_empty_clears() {
        let history = vec![message("history")];
        let mut prefix = vec![message("first"), message("second")];
        validate(&mut prefix).unwrap();
        let mut request = history.clone();
        prepend(&mut request, &prefix);
        assert_eq!(&request[..2], prefix.as_slice());
        assert_eq!(&request[2..], history.as_slice());
        let replacement = vec![message("new")];
        let mut request = history.clone();
        prepend(&mut request, &replacement);
        assert_eq!(request.len(), 2);
        let mut request = history.clone();
        prepend(&mut request, &[]);
        assert_eq!(request, history);
    }
    #[test]
    fn roles_text_and_stable_ids() {
        for role in ["user", "developer", "assistant"] {
            let mut items = vec![message("text")];
            let ResponseItem::Message { role: actual, .. } = &mut items[0] else {
                unreachable!()
            };
            *actual = role.into();
            validate(&mut items).unwrap();
            let before = items.clone();
            validate(&mut items).unwrap();
            assert_eq!(items, before);
        }
        let mut invalid = vec![message("text")];
        let ResponseItem::Message { role, .. } = &mut invalid[0] else {
            unreachable!()
        };
        *role = "system".into();
        assert!(validate(&mut invalid).is_err());
        assert!(validate(&mut [message(" ")]).is_err());
        validate(&mut []).unwrap();
        let mut duplicate_text = vec![message("same"), message("same")];
        validate(&mut duplicate_text).unwrap();
        assert_ne!(duplicate_text[0], duplicate_text[1]);
    }

    #[tokio::test]
    async fn atomic_replacement_invalid_update_clear_and_thread_isolation() {
        let first = RequestPrefix::default();
        let second = RequestPrefix::default();
        first.replace(vec![message("old")]).await.unwrap();
        let before = first.snapshot().await;
        assert!(
            first
                .replace(vec![message("valid"), message(" ")])
                .await
                .is_err()
        );
        assert_eq!(first.snapshot().await, before);
        assert!(second.snapshot().await.is_empty());
        first
            .replace(vec![message("new"), message("second")])
            .await
            .unwrap();
        assert_eq!(first.snapshot().await.len(), 2);
        assert_ne!(first.snapshot().await[0], before[0]);
        first.replace(vec![]).await.unwrap();
        assert!(first.snapshot().await.is_empty());
    }
}
