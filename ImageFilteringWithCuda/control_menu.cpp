#include "control_menu.h"

#include <iostream>
#include <string>
#include <conio.h>

void clearScreen() 
{
    #ifdef _WIN32
        system("cls"); // Windows
    #else
        system("clear"); // Linux/MacOS
    #endif
}

void displayFloatMenu(const std::vector<std::string>& options, int current) 
{
    clearScreen();
    std::cout << "Select predefined value, or enter new (Enter to apply):\n\n";
    for (size_t i = 0; i < options.size(); ++i) 
    {
        if (i == current) 
        {
            std::cout << " > " << options[i] << "\n";
        }
        else 
        {
            std::cout << "   " << options[i] << "\n";
        }
    }
}

float handleFloatSelection() 
{
    // Predefinied values
    std::vector<std::string> options = 
    {
        "1.0", 
        "2.5", 
        "5.0", 
        "new value..."
    };

    int current = 0; // first pick
    while (true) 
    {
        displayFloatMenu(options, current);

        char input = _getch(); // getting pressed button
        if (input == 'w' || input == 'W') // up
        { 
            current = (current > 0) ? current - 1 : options.size() - 1;
        }
        else if (input == 's' || input == 'S') // down
        {
            current = (current < options.size() - 1) ? current + 1 : 0;
        }
        else if (input == '\r') // Enter
        {
            if (current < 3) 
            {
                return std::stof(options[current]); // string -> float - conversion
            }
            else // new user value
            {
                clearScreen();
                std::cout << "new value (float): ";
                float userValue;
                std::cin >> userValue;
                return userValue;
            }
        }
    }
}

int xyOperatorMenu() 
{
    std::vector<std::string> subOptions = 
    {
        "Operator X",
        "Operator Y"
    };

    int current = 0;
    while (true) 
    {
        clearScreen();
        std::cout << "Select type:\n\n";
        for (size_t i = 0; i < subOptions.size(); ++i) {
            if (i == current) 
            {
                std::cout << " > " << subOptions[i] << "\n";
            }
            else 
            {
                std::cout << "   " << subOptions[i] << "\n";
            }
        }

        char input = _getch();
        if (input == 'w' || input == 'W') 
        {
            current = (current > 0) ? current - 1 : subOptions.size() - 1;
        }
        else if (input == 's' || input == 'S') 
        {
            current = (current < subOptions.size() - 1) ? current + 1 : 0;
        }
        else if (input == '\r') // Enter
        {
            clearScreen();
            std::cout << "Selected: " << subOptions[current] << "\n";
            return current;
            break;
        }
    }
}

int kernelSizeMenu() 
{
    std::vector<std::string> subOptions = {
        "3",
        "5",
        "7",
        "9"
    };

    int current = 0;
    while (true) 
    {
        clearScreen();
        std::cout << "Select num:\n\n";
        for (size_t i = 0; i < subOptions.size(); ++i) {
            if (i == current) 
            {
                std::cout << " > " << subOptions[i] << "\n";
            }
            else 
            {
                std::cout << "   " << subOptions[i] << "\n";
            }
        }

        char input = _getch();
        if (input == 'w' || input == 'W') 
        {
            current = (current > 0) ? current - 1 : subOptions.size() - 1;
        }
        else if (input == 's' || input == 'S') 
        {
            current = (current < subOptions.size() - 1) ? current + 1 : 0;
        }
        else if (input == '\r') // Enter
        {
            clearScreen();
            return std::stoi(subOptions[current]);
        }
    }
}

bool saveImageMenu()
{
    std::vector<std::string> subOptions = 
    {
        "No",
        "Yes"
    };

    int current = 0;
    while (true) 
    {
        clearScreen();
        std::cout << "Save image:\n\n";
        for (size_t i = 0; i < subOptions.size(); ++i) {
            if (i == current) 
            {
                std::cout << " > " << subOptions[i] << "\n";
            }
            else 
            {
                std::cout << "   " << subOptions[i] << "\n";
            }
        }

        char input = _getch();
        if (input == 'w' || input == 'W') 
        {
            current = (current > 0) ? current - 1 : subOptions.size() - 1;
        }
        else if (input == 's' || input == 'S') 
        {
            current = (current < subOptions.size() - 1) ? current + 1 : 0;
        }
        else if (input == '\r') // Enter
        { 
            clearScreen();
            std::cout << "save image: " << subOptions[current] << "\n";
            return current;
            break;
        }
    }
}