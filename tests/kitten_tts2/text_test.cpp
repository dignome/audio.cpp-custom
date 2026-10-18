#include "engine/community_models/kitten_tts2/text.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"
#include "engine/framework/text/text_normalization.h"

#include <iostream>
#include <stdexcept>

int main() {
    using engine::community_models::kitten_tts2::insert_sentence_pauses;
    using engine::community_models::kitten_tts2::normalize_prompt_text;
    using engine::text::TextChunkMode;
    int cases = 0;
    try {
        auto normalize_check = [&](const std::string & input, const std::string & language,
                                   const std::string & expected) {
            const auto actual = normalize_prompt_text(input, language);
            if (actual != expected) {
                std::cerr << "Language: " << language << "\nInput: " << input
                          << "\nExpected: " << expected << "\nActual: " << actual << '\n';
                throw std::runtime_error("unexpected text normalization");
            }
            ++cases;
        };
        normalize_check("I have 2 cats and 3.14 dollars.", "en", "I have two cats and three point one four dollars.");
        normalize_check("2 cats", " EN ", "two cats");
        normalize_check("I paid $1.00.", "en", engine::text::normalize_english_text("I paid $1.00."));
        normalize_check("[JoYfUl] I have 2 cats <PAUSE> and (((saved $2.50))).", "en",
            "[JoYfUl] I have two cats <PAUSE> and (((saved $2.50))).");
        normalize_check("<laugh> 1 cat <sigh> 2 cats", "en", "<laugh> one cat <sigh> two cats");
        normalize_check(u8"\uE000a\uE001 [sad] 2 cats", "en", u8"\uE000a\uE001 [sad] two cats");
        normalize_check(u8"\u6211\u67092\u53ea\u732b\u3002", "zh", u8"\u6211\u6709\u4e24\u53ea\u732b.");
        normalize_check(u8"\u4eca\u5929\u662f2024\u5e7410\u67081\u65e5\u3002", "zh",
            u8"\u4eca\u5929\u662f\u4e8c\u96f6\u4e8c\u56db\u5e74\u5341\u6708\u4e00\u65e5.");
        normalize_check("[joyful] 2 <pause> (((3)))", "zh", u8"[joyful] \u4e8c <pause> (((3)))");
        normalize_check("$50", "zh", u8"\u4e94\u5341\u7f8e\u5143");
        for (const auto * language : {"", "auto", "Auto", "fr", "ja", "en-US", "zh-CN"}) {
            normalize_check("  I paid $1.00.\n", language, "  I paid $1.00.\n");
            normalize_check(u8"\u6211\u67092\u53ea\u732b\u3002", language, u8"\u6211\u67092\u53ea\u732b\u3002");
        }
        std::string many_tags;
        for (int i = 0; i < 60; ++i) many_tags += "[sad] <pause> (((2))) ";
        normalize_check(many_tags + "2 cats", "en", many_tags + "two cats");
        normalize_check("", "en", "");
        normalize_check(" \r\n\t", "en", "");
        const auto normalized = normalize_prompt_text("I have 2 cats. Then I have 3 cats.", "en");
        const auto paused = insert_sentence_pauses(normalized);
        if (paused != "I have two cats. <pause> Then I have three cats.")
            throw std::runtime_error("normalization and sentence pauses do not compose");
        ++cases;
        const auto expanded = normalize_prompt_text("12345 12345 12345 12345", "en");
        const auto normalized_chunks = engine::text::split_text_chunks(expanded, 32, TextChunkMode::TagAware);
        if (normalized_chunks.size() < 2) throw std::runtime_error("expanded text was not chunked");
        for (const auto & chunk : normalized_chunks)
            if (engine::text::utf8_codepoint_count(chunk, "normalized chunk") > 32)
                throw std::runtime_error("normalized chunk exceeds character budget");
        ++cases;
        for (const auto & invalid : {std::string("\xE3\x80"), std::string("\xFF")}) {
            bool rejected = false;
            try { normalize_prompt_text(invalid, "en"); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("normalization accepted invalid UTF-8");
            ++cases;
        }
        auto pause_check = [&](const std::string & input, const std::string & expected) {
            const auto actual = insert_sentence_pauses(input);
            if (actual != expected) {
                std::cerr << "Input: " << input << "\nActual: " << actual << '\n';
                throw std::runtime_error("unexpected sentence pause insertion");
            }
            // Reprocessing must not add another pause at an existing boundary.
            if (insert_sentence_pauses(actual) != actual) throw std::runtime_error("duplicate pause insertion");
            ++cases;
        };
        pause_check("The sun comes into the room. A small bird sings near the window.",
            "The sun comes into the room. <pause> A small bird sings near the window.");
        pause_check("Dr. Smith paid 3.14 dollars. He left.", "Dr. Smith paid 3.14 dollars. <pause> He left.");
        pause_check("Mr. Jones met Prof. Smith. They spoke.", "Mr. Jones met Prof. Smith. <pause> They spoke.");
        pause_check("J. R. Smith lives in the U.S. today. We called.", "J. R. Smith lives in the U.S. today. <pause> We called.");
        pause_check("Use e.g. this example. Then stop.", "Use e.g. this example. <pause> Then stop.");
        pause_check("Version 1.2.3 is at example.com today. Open it.", "Version 1.2.3 is at example.com today. <pause> Open it.");
        pause_check("Wait... what happened? Really?! Yes!", "Wait... what happened? Really?! Yes!");
        pause_check("\"Hello.\" She smiled.", "\"Hello.\" <pause> She smiled.");
        pause_check("  Hello.  \tWorld. \n", "  Hello. <pause>  \tWorld. \n");
        pause_check("One line\r\nAnother line", "One line\r\nAnother line");
        pause_check("Hello.\nWelcome.", "Hello. <pause>\nWelcome.");
        pause_check("Hello.\n\nWelcome.", "Hello. <pause>\n\nWelcome.");
        pause_check("What? Really! Yes. Continue.", "What? Really! Yes. <pause> Continue.");
        pause_check("Wait... Then continue.", "Wait... Then continue.");
        pause_check("Hello. <laugh> Welcome.", "Hello. <pause> <laugh> Welcome.");
        pause_check("Hello.\t<PAUSE> Welcome.", "Hello.\t<PAUSE> Welcome.");
        pause_check(u8"\u201cHello.\u201d Welcome.", u8"\u201cHello.\u201d <pause> Welcome.");
        pause_check("(Hello.) Welcome.", "(Hello.) <pause> Welcome.");
        pause_check("  One sentence.  ", "  One sentence.  ");
        pause_check(" \r\n\t", " \r\n\t");
        pause_check("", "");
        pause_check("Hello. <pause> Welcome.", "Hello. <pause> Welcome.");
        pause_check("Hello. <PAUSE> Welcome.", "Hello. <PAUSE> Welcome.");
        pause_check("Hello.\n<pause>\nWelcome.", "Hello.\n<pause>\nWelcome.");
        pause_check("[JoYfUl] We won! I can (((hardly))) believe it.",
            "[JoYfUl] We won! I can (((hardly))) believe it.");
        pause_check("Keep (((both. together))) here. Continue.", "Keep (((both. together))) here. <pause> Continue.");
        pause_check("Keep [section 3.4] here. Continue.", "Keep [section 3.4] here. <pause> Continue.");
        pause_check("Hello! <laugh> Welcome.", "Hello! <laugh> Welcome.");
        pause_check("x < 5. Continue.", "x < 5. <pause> Continue.");
        pause_check(u8"\u4f60\u597d\u3002\u518d\u89c1\uff01", u8"\u4f60\u597d\u3002\u518d\u89c1\uff01");
        pause_check(u8"Le soleil entre. L\u2019oiseau chante.", u8"Le soleil entre. <pause> L\u2019oiseau chante.");
        pause_check(u8"Hola. \u00bfC\u00f3mo est\u00e1s?", u8"Hola. <pause> \u00bfC\u00f3mo est\u00e1s?");
        pause_check(u8"\u0645\u0631\u062d\u0628\u0627\u061f \u0646\u0639\u0645\u06d4", u8"\u0645\u0631\u062d\u0628\u0627\u061f \u0646\u0639\u0645\u06d4");
        pause_check(u8"\u0928\u092e\u0938\u094d\u0924\u0947\u0964 \u092b\u093f\u0930 \u092e\u093f\u0932\u0947\u0902\u0917\u0947\u0965", u8"\u0928\u092e\u0938\u094d\u0924\u0947\u0964 \u092b\u093f\u0930 \u092e\u093f\u0932\u0947\u0902\u0917\u0947\u0965");
        for (auto mode : {TextChunkMode::Default, TextChunkMode::TagAware, TextChunkMode::Japanese, TextChunkMode::Endline}) {
            const std::string original = "First sentence. Second sentence. Third sentence.";
            for (int64_t budget : {32, 1000}) {
                auto chunks = engine::text::split_text_chunks(original, budget, mode);
                const auto count = chunks.size();
                for (auto & chunk : chunks) {
                    const auto before = chunk;
                    chunk = insert_sentence_pauses(chunk);
                    auto restored = chunk;
                    for (size_t pos; (pos = restored.find(" <pause>")) != std::string::npos;)
                        restored.erase(pos, 8);
                    if (restored != before) throw std::runtime_error("pause insertion changed source text");
                }
                if (chunks.size() != count) throw std::runtime_error("pause insertion changed chunk count");
                ++cases;
            }
        }
        for (const auto & invalid : {std::string("\xE3\x80"), std::string("\xFF")}) {
            bool invalid_rejected = false;
            try { insert_sentence_pauses(invalid); }
            catch (const std::runtime_error &) { invalid_rejected = true; }
            if (!invalid_rejected) throw std::runtime_error("pause insertion accepted invalid UTF-8");
            ++cases;
        }
        std::cout << "PASS " << cases << " Kitten normalization, period pause, markup, Unicode and chunk budget cases\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
