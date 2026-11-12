#include <bits/stdc++.h>

using namespace std;

struct FFT{
    vector<double> coefs;
    vector<complex<double>> vals;
    const double pi = acos(-1);

    FFT(vector<double> cf){
        coefs = cf;
    }

    int get_log(int n){
        int i = 1, cnt = 0;
        while(i < n){
            i <<= 1;
            cnt++;
        }
        return cnt;
    }
    
    void print_fft(){
        for(auto& z : vals){
            cout << z << '\n';
        }
    }

    void bit_reversal(){

        for(int i = 0; i < vals.size(); i++){
            int rev = 0;
            for(int j = 0; j < get_log(vals.size()); j++){
                if((i >> j) & 1)
                    rev |= (1 << (get_log(vals.size()) - 1 - j));
            }

            if(i < rev)
            swap(vals[i], vals[rev]);
        }
    }

    void fft(){
        while(coefs.size() & (coefs.size() - 1))
            coefs.push_back(0);

        for(double& x : coefs){
            complex<double> z(x, 0);
            vals.push_back(z);
        }
        bit_reversal();

        int N = vals.size();
        int L = get_log(N), step = 1, power = 2;        
        
        for(int level = 0; level < L; level++){

            vector<complex<double>> eps;
            for(int i = 0; i < power; i++)
                eps.push_back(polar(1.0, -2.0 * pi * i/ power));


            for(int start = 0; start < N; start += step * 2){
                    
                    for(int k = 0; k < step; k++){
                        complex<double> a = vals[start + k], b = vals[start + k + step];

                        vals[start + k] = a + eps[(start + k) % power] * b;
                        vals[start + k + step] = a - eps[(start + k) % power] * b;
                    }
            }

            power *= 2;
            step *= 2;
        }
    }


};

int main(int argc, char** argv){
    vector<double> time_series;
    int N;

    if(!(cin >> N) || N < 0)
        return 1;
    for(int i = 0; i < N; i++){
        double x;
        if(!(cin >> x))
            return 1;
        time_series.push_back(x);
    }

    if(argc == 3 && string(argv[1]) == "--benchmark"){
        int repeats = stoi(argv[2]);
        if(repeats < 1)
            return 1;

        FFT warmup(time_series);
        warmup.fft();

        vector<double> times;
        double checksum = 0;
        for(int run = 0; run < repeats; run++){
            FFT solver(time_series);
            auto begin = chrono::steady_clock::now();
            solver.fft();
            auto end = chrono::steady_clock::now();

            times.push_back(chrono::duration<double>(end - begin).count());
            if(!solver.vals.empty())
                checksum += solver.vals[0].real();
        }

        sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        if(times.size() % 2 == 0)
            median = (median + times[times.size() / 2 - 1]) / 2;

        cout << setprecision(17) << median << '\n';
        if(checksum == numeric_limits<double>::infinity())
            cerr << checksum;
        return 0;
    }

    if(argc != 1)
        return 1;

    FFT solver(time_series);
    solver.fft();
    solver.print_fft();

    return 0;
}
