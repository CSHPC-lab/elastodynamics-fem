#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
int get_env_int(const char *name, int fallback)
{
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}

int local_rank()
{
    int lr = get_env_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (lr >= 0)
        return lr;
    lr = get_env_int("SLURM_LOCALID", -1);
    if (lr >= 0)
        return lr;
    return 0;
}

void sleep_us(int us)
{
    if (us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(us));
}

void busy_work_us(int us)
{
    if (us <= 0)
        return;
    const double t0 = MPI_Wtime();
    const double sec = us * 1.0e-6;
    volatile double x = 1.0;
    while (MPI_Wtime() - t0 < sec)
    {
        x = x * 1.0000000001 + 1.0;
        if (x > 1.0e100)
            x = 1.0;
    }
}

struct Options
{
    int iters = 1000;
    int warmup = 100;
    int skew_us = 0;
    int work_us = 0;
    int poll_chunk_us = 50;
};

Options parse_options(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; i++)
    {
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << name << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--iters") == 0)
            opt.iters = std::atoi(need_value("--iters"));
        else if (std::strcmp(argv[i], "--warmup") == 0)
            opt.warmup = std::atoi(need_value("--warmup"));
        else if (std::strcmp(argv[i], "--skew-us") == 0)
            opt.skew_us = std::atoi(need_value("--skew-us"));
        else if (std::strcmp(argv[i], "--work-us") == 0)
            opt.work_us = std::atoi(need_value("--work-us"));
        else if (std::strcmp(argv[i], "--poll-chunk-us") == 0)
            opt.poll_chunk_us = std::atoi(need_value("--poll-chunk-us"));
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                std::cout
                    << "Usage: mpi_allreduce_bench [options]\n"
                    << "  --iters N          measured iterations (default: 1000)\n"
                    << "  --warmup N         warmup iterations (default: 100)\n"
                    << "  --skew-us N        artificial per-local-rank arrival skew before each sync\n"
                    << "                     rank delay = local_rank * N usec (default: 0)\n"
                    << "  --work-us N        local work after MPI_Iallreduce issue (default: 0)\n"
                    << "  --poll-chunk-us N  polling chunk for Iallreduce+poll case (default: 50)\n";
            }
            MPI_Finalize();
            std::exit(0);
        }
    }
    return opt;
}

struct Result
{
    std::string name;
    int calls_per_iter = 0;
    double elapsed = 0.0;
    double barrier_elapsed = 0.0;
    double wait_elapsed = 0.0;
};

void arrival_skew(const Options &opt, int lr)
{
    sleep_us(opt.skew_us * lr);
}

template <class Fn>
double timed_loop(int warmup, int iters, Fn fn)
{
    for (int i = 0; i < warmup; i++)
        fn(i);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++)
        fn(i);
    return MPI_Wtime() - t0;
}

Result bench_baseline_3_scalar(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "current_3_scalar_allreduce";
    r.calls_per_iter = 3;
    double a = rank + 1.0, b = rank + 2.0, c = rank + 3.0;
    r.elapsed = timed_loop(opt.warmup, opt.iters, [&](int i) {
        a += 1.0e-12 * i;
        b += 1.0e-12 * i;
        c += 1.0e-12 * i;
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    });
    return r;
}

Result bench_barrier_diag_3_scalar(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "diagnostic_barrier_then_3_scalar";
    r.calls_per_iter = 3;
    double a = rank + 1.0, b = rank + 2.0, c = rank + 3.0;
    for (int i = 0; i < opt.warmup; i++)
    {
        arrival_skew(opt, lr);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrival_skew(opt, lr);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrival_skew(opt, lr);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double total0 = MPI_Wtime();
    for (int i = 0; i < opt.iters; i++)
    {
        double t0;
        arrival_skew(opt, lr);
        t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        r.barrier_elapsed += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        r.elapsed += MPI_Wtime() - t0;

        arrival_skew(opt, lr);
        t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        r.barrier_elapsed += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        r.elapsed += MPI_Wtime() - t0;

        arrival_skew(opt, lr);
        t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        r.barrier_elapsed += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        r.elapsed += MPI_Wtime() - t0;
    }
    r.wait_elapsed = MPI_Wtime() - total0;
    return r;
}

Result bench_combined_3(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "mitigation_1_allreduce_3_doubles";
    r.calls_per_iter = 1;
    double v[3] = {rank + 1.0, rank + 2.0, rank + 3.0};
    r.elapsed = timed_loop(opt.warmup, opt.iters, [&](int i) {
        v[0] += 1.0e-12 * i;
        v[1] += 1.0e-12 * i;
        v[2] += 1.0e-12 * i;
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, v, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    });
    return r;
}

Result bench_combined_2(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "mitigation_2_allreduces_pAp_plus_rnorm_rz";
    r.calls_per_iter = 2;
    double pAp = rank + 1.0;
    double rr[2] = {rank + 2.0, rank + 3.0};
    r.elapsed = timed_loop(opt.warmup, opt.iters, [&](int i) {
        pAp += 1.0e-12 * i;
        rr[0] += 1.0e-12 * i;
        rr[1] += 1.0e-12 * i;
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrival_skew(opt, lr);
        MPI_Allreduce(MPI_IN_PLACE, rr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    });
    return r;
}

Result bench_iallreduce_busy_3(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "experiment_3_iallreduce_busy_work";
    r.calls_per_iter = 3;
    double a = rank + 1.0, b = rank + 2.0, c = rank + 3.0;
    MPI_Request req;
    r.elapsed = timed_loop(opt.warmup, opt.iters, [&](int i) {
        a += 1.0e-12 * i;
        b += 1.0e-12 * i;
        c += 1.0e-12 * i;
        arrival_skew(opt, lr);
        MPI_Iallreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        busy_work_us(opt.work_us);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        arrival_skew(opt, lr);
        MPI_Iallreduce(MPI_IN_PLACE, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        busy_work_us(opt.work_us);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        arrival_skew(opt, lr);
        MPI_Iallreduce(MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        busy_work_us(opt.work_us);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    });
    return r;
}

Result bench_iallreduce_poll_3(const Options &opt, int rank, int lr)
{
    Result r;
    r.name = "experiment_3_iallreduce_poll_progress";
    r.calls_per_iter = 3;
    double a = rank + 1.0, b = rank + 2.0, c = rank + 3.0;
    auto one = [&](double &x) {
        MPI_Request req;
        arrival_skew(opt, lr);
        MPI_Iallreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        int done = 0;
        int remaining = opt.work_us;
        while (!done && remaining > 0)
        {
            const int chunk = std::min(opt.poll_chunk_us, remaining);
            busy_work_us(chunk);
            remaining -= chunk;
            MPI_Test(&req, &done, MPI_STATUS_IGNORE);
        }
        if (!done)
            MPI_Wait(&req, MPI_STATUS_IGNORE);
    };
    r.elapsed = timed_loop(opt.warmup, opt.iters, [&](int i) {
        a += 1.0e-12 * i;
        b += 1.0e-12 * i;
        c += 1.0e-12 * i;
        one(a);
        one(b);
        one(c);
    });
    return r;
}

void print_result(const Result &r, const Options &opt, int rank)
{
    double vals[3] = {r.elapsed, r.barrier_elapsed, r.wait_elapsed};
    double maxv[3] = {}, minv[3] = {}, sumv[3] = {};
    MPI_Reduce(vals, maxv, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(vals, minv, 3, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(vals, sumv, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank != 0)
        return;

    const double calls = static_cast<double>(opt.iters) * r.calls_per_iter;
    const double avg = sumv[0] / size;
    const double max_us_per_call = calls > 0.0 ? maxv[0] * 1.0e6 / calls : 0.0;
    std::cout << std::left << std::setw(42) << r.name
              << " total_max=" << std::right << std::setw(10) << std::fixed << std::setprecision(6) << maxv[0] << " s"
              << " avg=" << std::setw(10) << avg << " s"
              << " min=" << std::setw(10) << minv[0] << " s"
              << " max_per_call=" << std::setw(10) << std::setprecision(2) << max_us_per_call << " us";
    if (r.barrier_elapsed > 0.0 || r.wait_elapsed > 0.0)
    {
        const double barrier_us = calls > 0.0 ? maxv[1] * 1.0e6 / calls : 0.0;
        const double total_diag_us = calls > 0.0 ? maxv[2] * 1.0e6 / calls : 0.0;
        std::cout << " barrier_max=" << std::setw(10) << std::setprecision(6) << maxv[1] << " s"
                  << " barrier_per_call=" << std::setw(10) << std::setprecision(2) << barrier_us << " us"
                  << " diag_total_per_call=" << std::setw(10) << total_diag_us << " us";
    }
    std::cout << std::endl;
}
} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const Options opt = parse_options(argc, argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int lr = local_rank();

    if (rank == 0)
    {
        std::cout << "MPI AllReduce benchmark\n"
                  << "  ranks=" << size
                  << " iters=" << opt.iters
                  << " warmup=" << opt.warmup
                  << " skew_us_per_local_rank=" << opt.skew_us
                  << " iallreduce_work_us=" << opt.work_us
                  << " poll_chunk_us=" << opt.poll_chunk_us << std::endl;
    }
    std::cout << "rank=" << rank
              << " local_rank=" << lr
              << " host=" << (std::getenv("HOSTNAME") ? std::getenv("HOSTNAME") : "")
              << " CUDA_VISIBLE_DEVICES=" << (std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "")
              << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<Result> results;
    results.push_back(bench_baseline_3_scalar(opt, rank, lr));
    results.push_back(bench_barrier_diag_3_scalar(opt, rank, lr));
    results.push_back(bench_combined_3(opt, rank, lr));
    results.push_back(bench_combined_2(opt, rank, lr));
    results.push_back(bench_iallreduce_busy_3(opt, rank, lr));
    results.push_back(bench_iallreduce_poll_3(opt, rank, lr));

    if (rank == 0)
        std::cout << "\nResults (rank max is the main value to compare with PCG timing):\n";
    for (const Result &r : results)
        print_result(r, opt, rank);

    MPI_Finalize();
    return 0;
}
