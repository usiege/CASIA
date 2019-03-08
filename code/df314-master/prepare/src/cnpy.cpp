#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <stdint.h>
#include <stdexcept>

char big_endian_test()
{
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs)
{
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++)
        lhs.push_back(rhs[byte]);
    return lhs;
}

char map_type(const std::type_info& t)
{
    if (t == typeid(float))                     return 'f';
    if (t == typeid(double))                    return 'f';
    if (t == typeid(long double))               return 'f';

    if (t == typeid(int))                       return 'i';
    if (t == typeid(char))                      return 'i';
    if (t == typeid(short))                     return 'i';
    if (t == typeid(long))                      return 'i';
    if (t == typeid(long long))                 return 'i';

    if (t == typeid(unsigned char))             return 'u';
    if (t == typeid(unsigned short))            return 'u';
    if (t == typeid(unsigned long))             return 'u';
    if (t == typeid(unsigned long long))        return 'u';
    if (t == typeid(unsigned int))              return 'u';

    if (t == typeid(bool))                      return 'b';

    if (t == typeid(std::complex<float>))       return 'c';
    if (t == typeid(std::complex<double>))      return 'c';
    if (t == typeid(std::complex<long double>)) return 'c';

    return '?';
}

void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order)
{
    uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer + 6);
    uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer + 7);
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
    std::string header(reinterpret_cast<char*>(buffer + 9), header_len);

    size_t loc1, loc2;

    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    loc1 = header.find("(");
    loc2 = header.find(")");

    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    for (int i = 0; i < str_shape.size(); i++)
    {
        // skip the non-number character.
        if (str_shape[i] < '0' || str_shape[i] >'9')
        {
            i++;
            continue;
        }

        int t = 0;
        while(str_shape[i]>='0' && str_shape[i]<'9')
        {
            t = t * 10 + (str_shape[i] - '0');
            i++;
        }
        shape.push_back(t);
    }

    loc1 = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
    assert(littleEndian);

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order)
{
    char buffer[256], *desc;
    fgets(buffer, 256, fp);
    desc = buffer + 11;
    assert(desc[strlen(desc) - 1] == '\n');
    parse_npy_header((unsigned char*)buffer, word_size, shape, fortran_order);
}

void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset)
{
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread!");
    
    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no                 = *(uint16_t*) &footer[4];
    disk_start              = *(uint16_t*) &footer[6];
    nrecs_on_disk           = *(uint16_t*) &footer[8];
    nrecs                   = *(uint16_t*) &footer[10];
    global_header_size      = *(uint32_t*) &footer[12];
    global_header_offset    = *(uint32_t*) &footer[16];
    comment_len             = *(uint16_t*) &footer[20];

    assert(disk_no          == 0);
    assert(disk_start       == 0);
    assert(nrecs_on_disk    == nrecs);
    assert(comment_len      == 0);
}

NpyArray load_npy_file(FILE* fp)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);

    if (nread != arr.num_bytes())
    {
        char message[100];
        snprintf(message, 100, "load_npy_file: nread(%lu) is not equal num_bytes(%lu)!\n", nread, arr.num_bytes());
        throw std::runtime_error(message);
    }

    return arr;
}

NpyArray load_npz_file(FILE* fp, uint32_t compr_bytes, uint32_t uncompr_bytes)
{
    std::vector<unsigned char> buffer_compr(compr_bytes);
    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
    size_t nread = fread(&buffer_compr[0], 1, compr_bytes, fp);
    if(nread != compr_bytes)
        throw std::runtime_error("load_npz_file: failed fread!");

    int err;
    z_stream d_stream;

    d_stream.zalloc     = Z_NULL;
    d_stream.zfree      = Z_NULL;
    d_stream.opaque     = Z_NULL;
    d_stream.avail_in   = 0;
    d_stream.next_in    = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in   = compr_bytes;
    d_stream.next_in    = &buffer_compr[0];
    d_stream.avail_out  = uncompr_bytes;
    d_stream.next_out   = &buffer_uncompr[0];

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(&buffer_uncompr[0], word_size, shape, fortran_order);

    NpyArray array(shape, word_size, fortran_order);
    size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<unsigned char>(), &buffer_uncompr[0] + offset,array.num_bytes());

    return array;
}

NpyArray npy_load(std::string fname)
{
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npy_load: Unable to open file " + fname + "!");
    
    NpyArray arr = load_npy_file(fp);
    fclose(fp);
    return arr;
}

npz_t npz_load(std::string fname)
{
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npz_load: Error! Unable to open file " + fname + "!");

    npz_t arrays;  

    while (1)
    {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);
        if(headerres != 30)
            throw std::runtime_error("npz_load: failed fread header!");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string varname(name_len,' ');
        size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread name!");

        //erase the lagging .npy        
        varname.erase(varname.end() - 4, varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0], sizeof(char), extra_field_len, fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread extra field!");
        }

        uint16_t compr_method   = *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes    = *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes  = *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if(compr_method == 0) 
            arrays[varname] = load_npy_file(fp);
        else 
            arrays[varname] = load_npz_file(fp, compr_bytes, uncompr_bytes);
    }

    fclose(fp);
    return arrays;  
}

NpyArray npz_load(std::string fname, std::string varname)
{
    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) 
        throw std::runtime_error("npz_load: Unable to open file " + fname + "!");

    while(1)
    {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0], sizeof(char), 30, fp);
        if(header_res != 30)
            throw std::runtime_error("npz_load: failed fread header!");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);      
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread name!");
        
        //erase the lagging .npy
        vname.erase(vname.end()-4,vname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp, extra_field_len, SEEK_CUR); //skip past the extra field
        
        uint16_t compr_method   = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes    = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes  = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(vname == varname) {
            NpyArray array  = (compr_method == 0) ? 
                              load_npy_file(fp) : 
                              load_npz_file(fp, compr_bytes, uncompr_bytes);
            fclose(fp);
            return array;
        }
        else
        {
            //skip past the data
            uint32_t size = *(uint32_t*) &local_header[22];
            fseek(fp, size, SEEK_CUR);
        }
    }
}

