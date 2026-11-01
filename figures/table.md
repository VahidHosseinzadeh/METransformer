| model | seeds | test MSE t=16-25 (10k) | median | shared: t=16-25 | shared: t=100 | shared: MSE > 1e-2 | boundary vel. acc. | s/epoch |
|---|---|---|---|---|---|---|---|---|
| MET | 3 | 7.88e-04 ± 3.0e-06 | 1.21e-07 ± 6.1e-08 | 7.05e-04 ± 2.3e-06 | 7.92e-04 ± 1.8e-04 | 2.03 ± 0.00% | 98.92 ± 0.00% | 1024 |
| Transformer | 3 | 2.63e-02 ± 3.3e-04 | 2.17e-02 ± 3.3e-04 | 2.65e-02 ± 4.1e-04 | 1.35e-01 ± 2.1e-03 | 80.40 ± 1.00% | - | 546 |
| MELSTM (MELSTM paper) | 5 | 2.87e-4 ± 2.20e-4 | - | 2.30e-04 ± 1.7e-04 | 1.16e-03 ± 1.1e-03 (paper: 1.16e-3 ± 1.05e-3) | 0.44 ± 0.33% | - | 157 (A100) |
| FELSTM (MELSTM paper) | 5 | 1.54e-3 ± 0.42e-3 | - | 1.50e-03 ± 4.3e-04 | 8.83e-02 ± 6.6e-02 (paper: 8.83e-2 ± 6.58e-2) | 2.99 ± 1.27% | - | 918 (A100) |
| LSTM (MELSTM paper) | 5 | 3.26e-2 ± 0.05e-2 | - | 3.24e-02 ± 4.9e-04 | 2.00e-01 ± 1.3e-01 (paper: 2.01e-1 ± 1.34e-1) | 86.84 ± 0.52% | - | 77 (A100) |

Means ± s.d. over seeds. Test: first 10,000 test sequences (seed 123); MELSTM's column is its paper's, over 4 copies of the first 2,512 of them. Shared: the first 2,512 sequences of the 100-frame set (seed 42), the unique sequences of MELSTM's evaluation, scored by every model: mean MSE over t=16-25, MSE at t=100 (paper: over MELSTM's 10,000 rows, 4 copies of those), and the share of sequences whose t=16-25 MSE exceeds 1e-2. MET's velocities come from a parameter-free tracker, so its velocity accuracy is the same for every seed. s/epoch: training pass only, V100 for MET/Transformer, A100 for MELSTM's runs.
