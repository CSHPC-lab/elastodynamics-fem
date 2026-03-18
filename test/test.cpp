// インクルード
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
// 名前空間の使用
using namespace std;

// プロトタイプ宣言
void say_hello();
int add(int a, int b);
// オーバーロード
double add(double a, double b);
// テンプレート
template <typename T>
T multiply(T a, T b);

int main()
{
    std::string hello = "Please enter a string: ";
    double x = 10.34;
    const double pi = 3.14159;
    bool is_true = true;
    int IDs[5] = {1, 2, 3, 4, 5};
    // ポインタ
    double *x_ptr = &x;
    vector<int> vec = {1, 2, 3, 4, 5};

    say_hello();

    x += 5;

    std::cout << hello << std::endl;
    // stdがなくても名前空間のおかげで標準関数が使える
    cin >> hello;
    cout << "You entered: " << hello << endl;
    cout << "x: " << x << endl;
    cout << "x pointer: " << x_ptr << ", value: " << *x_ptr << endl;
    cout << "add(3, 4): " << add(3, 4) << endl;
    cout << "add(2.5, 3.5): " << add(2.5, 3.5) << endl;
    cout << "multiply(3, 4): " << multiply(3, 4) << endl;
    cout << "multiply(2.5, 3.5): " << multiply(2.5, 3.5) << endl;

    // if文
    if (is_true)
    {
        cout << "pi: " << pi << endl;
    }
    else if (!is_true && x > 10)
    {
        cout << "x is greater than 10" << endl;
    }
    else
    {
        cout << "x is not greater than 10" << endl;
    }

    // for文
    for (int i = 0; i < 5; i++)
    {
        if (i == 2)
        {
            cout << "Skipping ID: " << IDs[i] << endl;
            continue;
        }
        cout << "ID: " << IDs[i] << endl;
    }

    // ファイル操作
    ofstream output_file("test.txt");
    output_file << "Hello, this is a test file." << endl;

    return 0;
}

// 関数の定義
void say_hello()
{
    cout << "Hello, World!" << endl;
}

int add(int a, int b)
{
    return a + b;
}

double add(double a, double b)
{
    return a + b;
}

template <typename T>
T multiply(T a, T b)
{
    return a * b;
}