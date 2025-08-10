#pragma once

#include <string>
#include <unordered_map>

std::string readJsonFromFile(const std::string& jsonPath);
std::unordered_map<std::string, std::string> parseJson(const std::string& json);
std::unordered_map<std::string, std::string> getJsonContent(const std::string& jsonPath);
std::string toLowercase(std::string str);

bool operationTypeNCorrectFormat(const std::unordered_map<std::string, std::string>& jsonData);

bool getSobelDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData);

void getGaussianDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData, float& sigma, int& kernel_size);

bool getSaveDataFromJSON(const std::unordered_map<std::string, std::string>& jsonData, std::string& outputImagePath);