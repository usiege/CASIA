#include "cnpy.h"
#include "point.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

using namespace std;

#define DEFAULT_FOLDER  "data"

int main(int argc, char* argv[])
{
    char* root_folder;
    int device_id = 0;
    fprintf(stdout, "generate result...\n");

    if (argc >= 2)
        root_folder = argv[1];
    else
        root_folder = DEFAULT_FOLDER;

    if (argc >= 3)
        device_id = atoi(argv[2]);

    
    char fname[100];
    int count = 0;
    snprintf(fname, 100, "%s/pts", root_folder);
    vector<string> files = get_file_names(fname);

    fprintf(stdout, "Process start at %s, the number of file:%d\n", root_folder, files.size());
    prepare(root_folder ,files, device_id);
    fprintf(stdout, "\nProcess done!\n");

    return 0;
}
