// Minimal subset of nlohmann::json API used by this repo's demos.
// Note: This is NOT the full upstream library; it only supports:
// - objects with string keys
// - values: null, bool, number, string (arrays/objects parsed but minimally supported)
//
// If you want the full nlohmann/json, replace this file with the official single-header.

#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nlohmann {

class json {
public:
    enum class kind { null, boolean, number, string, object, array };

    using object_t = std::map<std::string, json>;
    using array_t  = std::vector<json>;

    json() : k_(kind::null) {}
    json(std::nullptr_t) : k_(kind::null) {}
    json(bool b) : k_(kind::boolean), b_(b) {}
    json(double d) : k_(kind::number), n_(d) {}
    json(int64_t i) : k_(kind::number), n_((double) i) {}
    json(std::string s) : k_(kind::string), s_(std::move(s)) {}
    json(const char * s) : k_(kind::string), s_(s ? s : "") {}
    json(object_t o) : k_(kind::object), o_(std::move(o)) {}
    json(array_t a) : k_(kind::array),  a_(std::move(a)) {}

    bool is_null()    const { return k_ == kind::null; }
    bool is_boolean() const { return k_ == kind::boolean; }
    bool is_number()  const { return k_ == kind::number; }
    bool is_string()  const { return k_ == kind::string; }
    bool is_object()  const { return k_ == kind::object; }
    bool is_array()   const { return k_ == kind::array; }

    bool contains(const std::string & key) const {
        if (!is_object()) return false;
        return o_.find(key) != o_.end();
    }

    const json & operator[](const std::string & key) const {
        static const json k_null{};
        if (!is_object()) return k_null;
        const auto it = o_.find(key);
        return it == o_.end() ? k_null : it->second;
    }

    template <typename T>
    T get() const;

    template <typename T>
    T value(const std::string & key, const T & def) const {
        if (!is_object()) return def;
        const auto it = o_.find(key);
        if (it == o_.end()) return def;
        try {
            return it->second.get<T>();
        } catch (...) {
            return def;
        }
    }

    static json parse(const std::string & text) {
        parser p(text);
        json v = p.parse_value();
        p.skip_ws();
        if (!p.eof()) {
            throw std::runtime_error("json: trailing characters");
        }
        return v;
    }

private:
    kind k_;
    bool b_ = false;
    double n_ = 0.0;
    std::string s_;
    object_t o_;
    array_t a_;

    class parser {
    public:
        explicit parser(const std::string & s) : s_(s) {}

        bool eof() const { return i_ >= s_.size(); }

        void skip_ws() {
            while (i_ < s_.size() && std::isspace((unsigned char) s_[i_])) ++i_;
        }

        json parse_value() {
            skip_ws();
            if (eof()) throw std::runtime_error("json: unexpected EOF");
            const char c = s_[i_];
            if (c == '{') return parse_object();
            if (c == '[') return parse_array();
            if (c == '"') return json(parse_string());
            if (c == 't' || c == 'f') return json(parse_bool());
            if (c == 'n') { parse_null(); return json(nullptr); }
            if (c == '-' || std::isdigit((unsigned char) c)) return json(parse_number());
            throw std::runtime_error("json: invalid value");
        }

    private:
        const std::string & s_;
        size_t i_ = 0;

        void expect(char c) {
            if (eof() || s_[i_] != c) throw std::runtime_error("json: unexpected character");
            ++i_;
        }

        std::string parse_string() {
            expect('"');
            std::string out;
            while (!eof()) {
                char c = s_[i_++];
                if (c == '"') break;
                if (c != '\\') {
                    out.push_back(c);
                    continue;
                }
                if (eof()) throw std::runtime_error("json: bad escape");
                const char e = s_[i_++];
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) throw std::runtime_error("json: bad \\u escape");
                        uint32_t cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = s_[i_++];
                            cp <<= 4;
                            if      (h >= '0' && h <= '9') cp |= (uint32_t) (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (uint32_t) (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (uint32_t) (h - 'A' + 10);
                            else throw std::runtime_error("json: bad \\u escape");
                        }
                        append_utf8(out, cp);
                    } break;
                    default: throw std::runtime_error("json: unsupported escape");
                }
            }
            return out;
        }

        static void append_utf8(std::string & out, uint32_t cp) {
            if (cp <= 0x7F) {
                out.push_back((char) cp);
            } else if (cp <= 0x7FF) {
                out.push_back((char) (0xC0 | ((cp >> 6) & 0x1F)));
                out.push_back((char) (0x80 | ( cp       & 0x3F)));
            } else if (cp <= 0xFFFF) {
                out.push_back((char) (0xE0 | ((cp >> 12) & 0x0F)));
                out.push_back((char) (0x80 | ((cp >>  6) & 0x3F)));
                out.push_back((char) (0x80 | ( cp        & 0x3F)));
            } else {
                out.push_back((char) (0xF0 | ((cp >> 18) & 0x07)));
                out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
                out.push_back((char) (0x80 | ((cp >>  6) & 0x3F)));
                out.push_back((char) (0x80 | ( cp        & 0x3F)));
            }
        }

        double parse_number() {
            const char * start = s_.c_str() + i_;
            char * end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start) throw std::runtime_error("json: bad number");
            i_ = (size_t) (end - s_.c_str());
            return v;
        }

        bool parse_bool() {
            if (s_.compare(i_, 4, "true") == 0) {
                i_ += 4;
                return true;
            }
            if (s_.compare(i_, 5, "false") == 0) {
                i_ += 5;
                return false;
            }
            throw std::runtime_error("json: bad boolean");
        }

        void parse_null() {
            if (s_.compare(i_, 4, "null") != 0) throw std::runtime_error("json: bad null");
            i_ += 4;
        }

        json parse_array() {
            expect('[');
            skip_ws();
            array_t a;
            if (!eof() && s_[i_] == ']') {
                ++i_;
                return json(std::move(a));
            }
            while (true) {
                a.push_back(parse_value());
                skip_ws();
                if (eof()) throw std::runtime_error("json: unexpected EOF in array");
                if (s_[i_] == ',') { ++i_; continue; }
                if (s_[i_] == ']') { ++i_; break; }
                throw std::runtime_error("json: expected ',' or ']'");
            }
            return json(std::move(a));
        }

        json parse_object() {
            expect('{');
            skip_ws();
            object_t o;
            if (!eof() && s_[i_] == '}') {
                ++i_;
                return json(std::move(o));
            }
            while (true) {
                skip_ws();
                if (eof() || s_[i_] != '"') throw std::runtime_error("json: expected string key");
                std::string key = parse_string();
                skip_ws();
                expect(':');
                json val = parse_value();
                o.emplace(std::move(key), std::move(val));
                skip_ws();
                if (eof()) throw std::runtime_error("json: unexpected EOF in object");
                if (s_[i_] == ',') { ++i_; continue; }
                if (s_[i_] == '}') { ++i_; break; }
                throw std::runtime_error("json: expected ',' or '}'");
            }
            return json(std::move(o));
        }
    };
};

template <>
inline std::string json::get<std::string>() const {
    if (!is_string()) throw std::runtime_error("json: not a string");
    return s_;
}

template <>
inline bool json::get<bool>() const {
    if (!is_boolean()) throw std::runtime_error("json: not a boolean");
    return b_;
}

template <>
inline int json::get<int>() const {
    if (!is_number()) throw std::runtime_error("json: not a number");
    return (int) n_;
}

template <>
inline int64_t json::get<int64_t>() const {
    if (!is_number()) throw std::runtime_error("json: not a number");
    return (int64_t) n_;
}

template <>
inline float json::get<float>() const {
    if (!is_number()) throw std::runtime_error("json: not a number");
    return (float) n_;
}

template <>
inline double json::get<double>() const {
    if (!is_number()) throw std::runtime_error("json: not a number");
    return n_;
}

} // namespace nlohmann

