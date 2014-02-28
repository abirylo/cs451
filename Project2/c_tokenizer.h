/* c_tokenizer.h */

#pragma once
#ifndef C_TOKENIZER_H
#define C_TOKENIZER_H

typedef struct
{
    char*       s;
    const char* delimiters;
    char*       current;
    char*       next;
    int         is_ignore_empties;
}
    tokenizer_t;

enum { TOKENIZER_EMPTIES_OK, TOKENIZER_NO_EMPTIES };

tokenizer_t tokenizer( const char* s, const char* delimiters, int empties );
const char* free_tokenizer( tokenizer_t* tokenizer );
const char* tokenize( tokenizer_t* tokenizer );

#endif
