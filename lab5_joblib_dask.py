import time
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from dask.distributed import Client, LocalCluster

def run_test(name, n_iter, backend_type, X_train, y_train, param_dist, results, use_scatter=True):
    rf = RandomForestClassifier(random_state=42)
    start_time = time.time()
    
    if backend_type == 'native':
        search = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
    else:
        scatter_list = [X_train, y_train] if use_scatter else []
        with joblib.parallel_backend('dask', scatter=scatter_list):
            search = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            
    elapsed = time.time() - start_time
    results.append({
        "Configuration": name,
        "Iterations": n_iter,
        "Time_seconds": round(elapsed, 2),
        "CV_Score": round(search.best_score_, 4)
    })
    print(f"Done: {name} executed in {elapsed:.2f} seconds. Score: {search.best_score_:.4f}")

def main():
    X, y = make_classification(
        n_samples=20000, n_features=20, n_informative=15,
        n_redundant=3, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
    }

    results = []

    print("Starting native joblib test")
    run_test("Native Joblib loky", 30, 'native', X_train, y_train, param_dist, results)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)

    print("\nDask cluster initialized")
    print(f"Dashboard link: {client.dashboard_link}")
    print("Open the link above in Safari and click Task Stream at the top menu to make a screenshot during the test")

    time.sleep(5)

    print("\nStarting Dask experiments")
    run_test("Joblib Dask 2 workers", 30, 'dask', X_train, y_train, param_dist, results, use_scatter=True)

    client.cluster.scale(1)
    time.sleep(3)
    run_test("Joblib Dask 1 worker", 30, 'dask', X_train, y_train, param_dist, results, use_scatter=True)

    client.cluster.scale(2)
    time.sleep(3)

    run_test("Joblib Dask No scatter", 30, 'dask', X_train, y_train, param_dist, results, use_scatter=False)
    run_test("Joblib Dask 60 iterations", 60, 'dask', X_train, y_train, param_dist, results, use_scatter=True)
    run_test("Native Joblib 60 iterations", 60, 'native', X_train, y_train, param_dist, results)

    client.close()
    cluster.close()

    print("\n=== FINAL RESULTS TABLE ===")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()
