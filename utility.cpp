#include "utility.h"

#include <iostream>
#include <fstream>
#include <cstdio>

Utility::Utility()
{
}

bool Utility::delFiles(const std::vector<std::string>& files)
{
    std::vector<std::string>::const_iterator cit = files.begin();
    while (cit != files.end())
    {
        std::ifstream file((*cit).c_str());
        if (file.good())
        {
            if (remove((*cit).c_str())!=0)
            {
                std::cerr << "Error deleting existing EM modele file (" << *cit << ")." << std::endl;
            }
        }
        file.close();
        cit++;
    }
}
