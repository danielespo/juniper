/* This heap was blatantly taken from MiniSAT by the EasySAT git, which deleted their copyright notice */
// I was disappointed to see they omitted the copyright notice from the original file, so I am attaching it below:

/*******************************************************************************************[heap.hpp]
Copyright (c) 2006-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/
#include <vector>
#include <fstream>
#define left(x) (x << 1 | 1)
#define right(x) ((x + 1) << 1)
#define father(x) ((x - 1) >> 1)

template<class Comp>
class Heap {
    Comp lt;
    std::vector<int> heap;
    std::vector<int> pos;
    
    void up(int v) {
        int x = heap[v], p = father(v);
        while (v && lt(x, heap[p])) {
            heap[v] = heap[p], pos[heap[p]] = v;
            v = p, p = father(p);
        }
        heap[v] = x, pos[x] = v;
    }

    void down(int v) {
        int x = heap[v];
        while (left(v) < (int)heap.size()){
            int child = right(v) < (int)heap.size() && lt(heap[right(v)], heap[left(v)]) ? right(v) : left(v);
            if (!lt(heap[child], x)) break;
            heap[v] = heap[child], pos[heap[v]] = v, v = child;
        }
        heap[v] = x, pos[x] = v;
    }

public:
    void setComp   (Comp c)              { lt = c; }
    bool empty     ()              const { return heap.size() == 0; }
    bool inHeap    (int n)         const { return n < (int)pos.size() && pos[n] >= 0; }
    void update    (int x)               { up(pos[x]); }

    void insert(int x) {
        if ((int)pos.size() < x + 1) 
            pos.resize(x + 1, -1);
        pos[x] = heap.size();
        heap.push_back(x);
        up(pos[x]); 
    }

    int pop() {
        int x = heap[0];
        heap[0] = heap.back();
        pos[heap[0]] = 0, pos[x] = -1;
        heap.pop_back();
        if (heap.size() > 1) down(0);
        return x; 
    }
};