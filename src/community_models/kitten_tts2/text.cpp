#include "engine/community_models/kitten_tts2/text.h"

#include "engine/framework/text/utf8.h"
#include "engine/framework/text/chinese_normalization.h"
#include "engine/framework/text/text_normalization.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <utility>
#include <vector>

namespace engine::community_models::kitten_tts2 {
namespace {
bool space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

size_t match_any(std::string_view value, size_t pos, std::initializer_list<std::string_view> items) {
    for (auto item : items) if (value.substr(pos, item.size()) == item) return item.size();
    return 0;
}

bool abbreviation(std::string_view text, size_t dot) {
    size_t begin = dot;
    while (begin && (std::isalpha(static_cast<unsigned char>(text[begin - 1])) || text[begin - 1] == '.')) --begin;
    std::string word(text.substr(begin, dot - begin));
    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (word.size() == 1 || word.find('.') != std::string::npos) return true;
    for (auto item : {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "no"})
        if (word == item) return true;
    return false;
}

size_t terminal_width(std::string_view value, size_t pos) {
    return match_any(value, pos, {".", "!", "?", u8"\u3002", u8"\uff01", u8"\uff1f", u8"\u061f", u8"\u06d4", u8"\u0964", u8"\u0965"});
}

using SentenceRange = std::pair<size_t, size_t>;
std::vector<SentenceRange> sentence_ranges(const std::string & value) {
    std::vector<SentenceRange> result;
    size_t begin = 0;
    auto emit = [&](size_t end) {
        size_t first = begin, last = end;
        while (first < last && space(value[first])) ++first;
        while (last > first && space(value[last - 1])) --last;
        if (first < last) result.emplace_back(first, last);
        begin = end;
    };
    for (size_t pos = 0; pos < value.size();) {
        // Keep bracketed text, events and emphasis together, including punctuation.
        const auto open = value.compare(pos, 3, "(((") == 0 ? "(((" :
            value[pos] == '[' ? "[" : value[pos] == '<' ? "<" : "";
        if (*open) {
            const std::string_view close = *open == '(' ? ")))" : *open == '[' ? "]" : ">";
            const size_t end = value.find(close, pos + std::char_traits<char>::length(open));
            if (end != std::string::npos) { pos = end + close.size(); continue; }
        }
        if (value[pos] != '.') { ++pos; continue; }
        size_t end = pos + 1;
        while (const auto next = terminal_width(value, end)) end += next;
        // Decimal points and abbreviations are not sentence endings. Preserve ellipses.
        if (abbreviation(value, pos) ||
            (end - pos > 1 && value.find_first_not_of('.', pos) >= end)) { pos = end; continue; }
        while (const auto close = match_any(value, end, {"\"", "'", ")", "]", u8"\u201d", u8"\u2019", u8"\u00bb", u8"\u300d", u8"\u300f", u8"\uff09"})) end += close;
        if (end < value.size() && !space(value[end])) { pos = end; continue; }
        emit(end);
        pos = end;
    }
    emit(value.size());
    return result;
}
}  // namespace

std::string normalize_prompt_text(std::string_view input, std::string_view language) {
    text::utf8_codepoint_count(input, "Kitten normalization text");
    std::string selected = text::collapse_ascii_whitespace(language);
    std::transform(selected.begin(), selected.end(), selected.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (selected != "en" && selected != "zh") return std::string(input);

    // Use digit-free, collision-free placeholders: both shared normalizers
    // expand digits, including digits inside naive numbered placeholders.
    const std::string source(input);
    std::string prefix = u8"\uE000";
    while (source.find(prefix) != std::string::npos) prefix += u8"\uE000";
    static const std::regex markup(
        R"(\[(mundane|nervous|tender|angry|excited|stern|sad|contemplative|surprised|joyful)\]|<(pause|sigh|gasp|laugh|giggle|sob|scoff|growl|um|gulp)>|\(\(\([^()\r\n]*\)\)\))",
        std::regex::icase);
    std::vector<std::pair<std::string, std::string>> saved;
    std::string protected_text;
    size_t copied = 0;
    for (std::sregex_iterator it(source.begin(), source.end(), markup), end; it != end; ++it) {
        std::string key;
        size_t index = saved.size();
        do { key += static_cast<char>('a' + index % 26); index /= 26; } while (index);
        key = prefix + key + u8"\uE001";
        protected_text.append(source, copied, static_cast<size_t>(it->position()) - copied);
        protected_text += key;
        saved.emplace_back(key, it->str());
        copied = static_cast<size_t>(it->position() + it->length());
    }
    protected_text.append(source, copied, std::string::npos);
    auto result = selected == "zh"
        ? text::normalize_chinese_text(protected_text)
        : text::normalize_english_text(protected_text);
    for (const auto & [key, value] : saved) result = text::replace_all(std::move(result), key, value);
    return result;
}

std::string insert_sentence_pauses(std::string_view input) {
    text::utf8_codepoint_count(input, "Kitten sentence text");
    const std::string source(input);
    const auto ranges = sentence_ranges(source);
    auto pause_at = [&](size_t pos) {
        if (pos + 7 > source.size()) return false;
        for (size_t i = 0; i < 7; ++i)
            if (std::tolower(static_cast<unsigned char>(source[pos + i])) != "<pause>"[i]) return false;
        return true;
    };
    std::string result;
    size_t copied = 0;
    for (size_t i = 1; i < ranges.size(); ++i) {
        const auto end = ranges[i - 1].second;
        // Keep explicit pause events, including ones on a separate line.
        if (pause_at(ranges[i].first) ||
            (end >= ranges[i - 1].first + 7 && pause_at(end - 7))) continue;
        result.append(source, copied, end - copied);
        result += " <pause>";
        if (!space(source[end])) result += ' ';
        copied = end;
    }
    result.append(source, copied, source.size() - copied);
    return result;
}

}  // namespace engine::community_models::kitten_tts2
