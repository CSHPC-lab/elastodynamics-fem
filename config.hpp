#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

struct Config
{
    std::unordered_map<std::string, std::string> data;

    void load(const std::string &path)
    {
        std::ifstream f(path);
        if (!f.is_open())
        {
            std::cerr << "ERROR: cannot open config file: " << path << std::endl;
            std::exit(1);
        }
        std::string line;
        while (std::getline(f, line))
        {
            // skip empty lines and comments
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream iss(line);
            std::string key, val;
            if (iss >> key >> val)
            {
                data[key] = val;
            }
        }
    }

    double get_double(const std::string &key) const { return std::stod(data.at(key)); }
    int get_int(const std::string &key) const { return std::stoi(data.at(key)); }
    std::string get_string(const std::string &key) const { return data.at(key); }
    bool get_bool(const std::string &key) const { return data.at(key) == "true"; }

    bool has(const std::string &key) const { return data.count(key) > 0; }
};
