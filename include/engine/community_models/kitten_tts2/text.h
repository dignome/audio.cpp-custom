#pragma once

#include <string>
#include <string_view>

namespace engine::community_models::kitten_tts2 {

// Optional written-to-spoken preprocessing before framework chunking. Only
// explicit en/zh language codes enable it; empty, auto and other languages
// preserve the input. Expression markup is kept verbatim.
std::string normalize_prompt_text(std::string_view input, std::string_view language = {});

// Apply after framework chunking. Preserve source text and add events only
// after sentence-ending ASCII periods within a chunk (following any closing
// quotes/brackets), without appending a trailing pause.
std::string insert_sentence_pauses(std::string_view input);

}  // namespace engine::community_models::kitten_tts2
