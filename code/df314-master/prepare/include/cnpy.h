#ifndef __FILE_OPT_H__
#define __FILE_OPT_H__

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cstring>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <numeric>

struct NpyArray
{
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order):
        shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
    {
        num_vals = 1;
        for (size_t i = 0; i < shape.size(); i++)
            num_vals *= shape[i];
        
        data_holder = std::shared_ptr<std::vector<char>>(
            new std::vector<char>(num_vals * word_size));
    }

    NpyArray(): shape(0), word_size(0), fortran_order(0), num_vals(0)
    { }

    template<typename T>
    T* data() const
    {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    std::vector<T> as_vec() const
    {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const
    {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

NpyArray npy_load(std::string fname);
npz_t npz_load(std::string fname);
NpyArray npz_load(std::string fname, std::string varname);

char big_endian_test();
char map_type(const std::type_info& t);
void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset);


std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs)
{
    for (size_t byte = 0; byte < sizeof(T); byte++)
    {
        char val = *((char*)&rhs + byte);
        lhs.push_back(val);
    }
}

template<typename T> std::vector<char> create_npy_header(const std::vector<size_t>& shape)
{
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += big_endian_test();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++)
    {
        dict += std::to_string(shape[i]);

        if (i == 0 || i < shape.size() - 1)
            dict += ", ";
    }
    dict += "), }";

    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char) 0x93;
    header += "NUMPY";
    header += (char) 0x01;
    header += (char) 0x00;
    header += (uint16_t) dict.size();
    header.insert(header.end(), dict.begin(), dict.end());
    
    return header;
}

template<typename T> void npy_save(std::string fname,
                                   const std::vector<T>& data,
                                   std::string mode)
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npy_save(fname, &data[0], shape, mode);
}

template<typename T> void npy_save(std::string fname,
                                   const T* data,
                                   const std::vector<size_t> shape,
                                   std::string mode)
{
    FILE* fp = NULL;
    std::vector<size_t> true_data_shape;

    if (mode == "a")
        fp = fopen(fname.c_str(), "r+b");
    
    if (fp)
    {
        size_t word_size;
        bool fortran_order;
        parse_npy_header(fp, word_size, true_data_shape, fortran_order);
        assert(!fortran_order);

        if (word_size != sizeof(T))
        {
            std::cout << "npy error: " << fname << " has word size " << word_size;
            std::cout << " but npy_save appending data sized " << sizeof(T) << '\n';
            assert(word_size == sizeof(T));
        }

        if (true_data_shape.size() != shape.size())
        {
            std::cout << "npy error: npy_save attempting to append misdimensioned data to " << fname << '\n';
            assert(true_data_shape.size() != shape.size());
        }

        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] != true_data_shape[i])
            {
                std::cout << "npy error: npy_save attempting to append misshaped data to " << fname << '\n';
                assert(shape[i] != true_data_shape[i]);
            }
        }

        true_data_shape[0] += shape[0];
    }
    else
    {
        fp = fopen(fname.c_str(), "wb");
        true_data_shape = shape;
    }

    std::vector<char> header = create_npy_header<T>(true_data_shape);
    size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    fseek(fp, 0, SEEK_SET);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
}

template<typename T> void npz_save(std::string zipname,
                                   std::string fname,
                                   const std::vector<T>& data,
                                   std::string mode)
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npz_save(zipname, fname, &data[0], shape, mode);
}

template<typename T> void npz_save(std::string zipname,
                                   std::string fname,
                                   const T* data,
                                   const std::vector<size_t>& shape,
                                   std::string mode)
{
    fname += ".npy";

    FILE* fp = NULL;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    std::vector<char> npy_header = create_npy_header<T>(shape);

    std::vector<char> global_header;
    std::vector<char> local_header;
    std::vector<char> footer;

    if (mode == "a")
        fp = fopen(zipname.c_str(), "r+b");
    
    if (fp)
    {
        size_t global_header_size;
        parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
        fseek(fp, global_header_offset, SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(&global_header[0], sizeof(char), global_header_size, fp);
        if (res != global_header_size)
            throw std::runtime_error("npz_save: header read error while adding to existing zip!");
        
        fseek(fp, global_header_offset, SEEK_SET);
    }
    else
    {
        fp = fopen(zipname.c_str(), "wb");
    }



    size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    size_t nbytes = nels * sizeof(T) + npy_header.size();

    uint32_t crc = crc32(0L, (uint8_t*)&npy_header[0], npy_header.size());
    crc = crc32(crc, (uint8_t*)data, nels * sizeof(T));

    /**
     * zip local file header format
     * 0    4    Local file header signature = 0x04034b50 (read as a little-endian number)
     * 4    2    Version needed to extract (minimum)
     * 6    2    General purpose bit flag
     * 8    2    Compression method
     * 10   2    File last modification time
     * 12   2    File last modification date
     * 14   4    CRC-32
     * 18   4    Compressed size
     * 22   4    Uncompressed size
     * 26   2    File name length (n)
     * 28   2    Extra field length (m)
     * 30   n    File name
     * 30+n m    Extra field
     **/
    local_header += (uint16_t) 0x4b50;
    local_header += (uint16_t) 0x0403;
    local_header += (uint16_t) 20;
    local_header += (uint16_t) 0;
    local_header += (uint16_t) 0;
    local_header += (uint16_t) 0;
    local_header += (uint16_t) 0;
    local_header += (uint32_t) crc;
    local_header += (uint32_t) nbytes;
    local_header += (uint32_t) nbytes;
    local_header += (uint16_t) fname.size();
    local_header += (uint16_t) 0;
    local_header += fname;

    /**
     * Central directory file header
     * Offset   Bytes   Description
     * 0        4       Central directory file header signature = 0x02014b50
     * 4        2       Version made by
     * 6        2       Version needed to extract (minimum)
     * 8        2       General purpose bit flag
     * 10       2       Compression method
     * 12       2       File last modification time
     * 14       2       File last modification date
     * 16       4       CRC-32
     * 20       4       Compressed size
     * 24       4       Uncompressed size
     * 28       2       File name length (n)
     * 30       2       Extra field length (m)
     * 32       2       File comment length (k)
     * 34       2       Disk number where file starts
     * 36       2       Internal file attributes
     * 38       4       External file attributes
     * 42       4       Relative offset of local file header.
     * 46       n       File name
     * 46+n     m       Extra field
     * 46+n+m   k       File comment
     **/
    global_header += (uint16_t) 0x4b50;
    global_header += (uint16_t) 0x0201;
    global_header += (uint16_t) 20;
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    global_header += (uint16_t) 0;
    global_header += (uint16_t) 0;
    global_header += (uint16_t) 0;
    global_header += (uint32_t) 0;
    global_header += (uint32_t) global_header_offset;
    global_header += fname;

    /**
     * End of central directory record (EOCD)
     * offset  Bytes    Description
     *  0       4     End of central directory signature = 0x06054b50
     *  4       2     Number of this disk
     *  6       2     Disk where central directory starts
     *  8       2     Number of central directory records on this disk
     *  10      2     Total number of central directory records
     *  12      4     Size of central directory (bytes)
     *  16      4     Offset of start of central directory, relative to start of archive
     *  20      2     Comment length (n)
     *  22      n     Comment
     **/
    footer += (uint16_t) 0x4b50;
    footer += (uint16_t) 0x0605;
    footer += (uint16_t) 0;
    footer += (uint16_t) 0;
    footer += (uint16_t) (nrecs + 1);
    footer += (uint16_t) (nrecs + 1);
    footer += (uint32_t) global_header.size();
    footer += (uint32_t) (global_header_offset + nbytes + local_header.size());
    footer += (uint16_t) 0;

    fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
    fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
    fwrite(data, sizeof(T), nels, fp);
    fwrite(&global_header[0], sizeof(char), global_header.size(), fp);
    fwrite(&footer[0], sizeof(char), footer.size(), fp);
    fclose(fp);
}                           

#endif