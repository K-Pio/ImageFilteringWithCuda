#pragma once

#include <vector>
#include <string>

void clearScreen();

void displayFloatMenu(const std::vector<std::string>& options, int current);
float handleFloatSelection();

int xyOperatorMenu();

int kernelSizeMenu();

bool saveImageMenu();