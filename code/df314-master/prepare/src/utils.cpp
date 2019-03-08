#include "utils.h"
#include "cnpy.h"

#include <dirent.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>

const std::vector<size_t> SHAPE = { 64, 4000, 4 };

void save_result(const char* fname, const double* results)
{
    std::string filename(fname);
    npy_save<double>(filename, results, SHAPE, "wb");
}

std::vector<std::string> get_file_names(const std::string &dir)
{
    std::vector<std::string> files;
    std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR *dir) { dir &&closedir(dir); });
    struct dirent *dirent_ptr;
    if (!directory_ptr)
    {
        std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
        return files;
    }

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr)
    {
        if (!strcmp(dirent_ptr->d_name, ".") || !strcmp(dirent_ptr->d_name, ".."))
            continue;

        files.push_back(std::string(dirent_ptr->d_name));
    }
    return files;
}
