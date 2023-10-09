# Reproduction of results

Install modules:

     pip install -U -r requirements.txt

Downloading datasets:

     python data/loader.py

Running the experiment:

     python runner.py --dataset {*dataset name*} --method {*method name*} --max-iter {*number of iterations*} --dir {*directory for results*} --trials {*number of trials*} --n_jobs {*the number of worker processes to use*}

`runner.py` script parameters:

1. --dataset – one or more from the list:

     (`balance`, `bank-marketing`, `banknote`, `breast-cancer`, `car-evaluation`, `cnae9`, `credit-approval`,
      `digits`, `ecoli`, `parkinsons`, `semeion`, `statlog-segmentation`, `wilt`, `zoo`)

2. --method – either `svc`, or `xgb`, or `mlp`
3. --max-iter – number of iterations
4. --dir – directory in which tables with results will be saved (by default this will be the `result` folder)
5. --trials – the number of trials in non-deterministic algorithms (`hyperopt`, `optuna`)
6. -n_jobs – the number of worker processes to use


## Launch example

We run the `svc` method with the `breast-cancer` and `zoo` datasets, the maximum number of iterations is `200`, trials with non-deterministic algorithms are `10`, the number of worker processes to use is `12`.

     python runner.py --dataset breast-cancer zoo --method svc --max-iter 200 --trials 10, --n-jobs 12

Once completed, the script will create two tables with the resulting metrics (`result/metrics.csv`) and times (`result/times.csv`). If the algorithm is non-deterministic, the table contains the mean with standard deviation.
