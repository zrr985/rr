#include "gcd.h"
int gcd(int a,int b){
    int t=a,r;
    if(a<b){
        a=b;
        b=t;
    }
    r=a%b;
    if(r==0) return b;
    return gcd(b,r);
}
