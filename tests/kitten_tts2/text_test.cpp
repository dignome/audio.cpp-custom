#include "engine/community_models/kitten_tts2/text.h"
#include "engine/framework/text/chunking.h"

#include <iostream>
#include <stdexcept>

int main() {
    using engine::community_models::kitten_tts2::insert_sentence_pauses;
    using engine::text::TextChunkMode;
    int cases = 0;
    try {
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
        std::cout << "PASS " << cases << " Kitten period pause, markup, Unicode and chunk budget cases\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
