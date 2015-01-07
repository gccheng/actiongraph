#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <string>

class Utility
{
public:
    Utility();

public:
    static bool delFiles(const std::vector<std::string>& files);
};

#endif // UTILITY_H
