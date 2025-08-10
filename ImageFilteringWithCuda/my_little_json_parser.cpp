#include "my_little_json_parser.h"

#include <iostream>
#include <fstream>
#include <algorithm>

std::string toLowercase(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

// JSON extraction
std::string readJsonFromFile(const std::string& jsonPath)
{
    std::string line, json;
    std::ifstream file(jsonPath);

    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            json += line;
        }
    }
    else
    {
        std::cerr << "Faild to read file: " << jsonPath << std::endl;
    }
    return json;
}
std::unordered_map<std::string, std::string> parseJson(const std::string& json)
{
    std::unordered_map<std::string, std::string> result;
    size_t pos = 0;

    while ((pos = json.find('"', pos)) != std::string::npos) 
    {
        size_t keyStart = pos + 1;
        size_t keyEnd = json.find('"', keyStart);
        std::string key = json.substr(keyStart, keyEnd - keyStart);

        size_t colonPos = json.find(':', keyEnd);
        size_t valueStart = json.find('"', colonPos) + 1;
        size_t valueEnd = json.find('"', valueStart);
        std::string value = json.substr(valueStart, valueEnd - valueStart);

        result[key] = value;
        pos = valueEnd + 1;
    }

    return result;
}
std::unordered_map<std::string, std::string> getJsonContent(const std::string& jsonPath)
{
    std::string json = readJsonFromFile(jsonPath);
    return parseJson(json);
}

// operation on JSON data
bool operationTypeNCorrectFormat(const std::unordered_map<std::string, std::string>& jsonData)
{
    bool gb;
    std::string oprType;
    if (jsonData.find("type") != jsonData.end())
    {
        oprType = jsonData.at("type");
    }
    else
    {
        std::cerr << "JSON file is not correct" << std::endl;
        exit(1);
    }
    if (jsonData.find("image path") == jsonData.end())
    {
        std::cerr << "JSON file is not correct" << std::endl;
        exit(1);
    }
    
    if (toLowercase(oprType) == "gaussian blur")
    {
        gb = true;
        
        if (jsonData.find("sigma") == jsonData.end())
        {
            std::cerr << "JSON file is not correct" << std::endl;
            exit(1);
        }
        if (jsonData.find("kernel size") == jsonData.end())
        {
            std::cerr << "JSON file is not correct" << std::endl;
            exit(1);
        }
    }
    else if (toLowercase(oprType) == "sobel operator")
    {
        gb = false;
        if (jsonData.find("operator type") == jsonData.end())
        {
            std::cerr << "JSON file is not correct" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "JSON file is not correct" << std::endl;
        exit(1);
    }
    
    return gb;
}

bool getSobelDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData)
{
    const std::string oprSize = jsonData.at("operator type");

    if (oprSize == "X")
    {
        return true;
    }
    else
    {
        return false;
    }
}

void getGaussianDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData, float& sigma, int& kernel_size)
{
    sigma = stof(jsonData.at("sigma"));
    kernel_size = stoi(jsonData.at("kernel size"));
}

bool getSaveDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData, std::string& outputImagePath)
{
    if (jsonData.find("save image") != jsonData.end())
    {
        outputImagePath = jsonData.at("save image");
        return true;
    }
    else
    {
        return false;
    }
}