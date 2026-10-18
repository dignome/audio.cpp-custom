#pragma once

#include <string>
#include <string_view>

namespace engine::community_models::kitten_tts2 {

// Apply after framework chunking. Preserve source text and add events only
// after sentence-ending ASCII periods within a chunk (following any closing
// quotes/brackets), without appending a trailing pause.
std::string insert_sentence_pauses(std::string_view input);

}  // namespace engine::community_models::kitten_tts2
