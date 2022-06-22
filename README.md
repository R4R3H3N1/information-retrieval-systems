# information-retrieval-systems
Implementation of tf-idf model and LSI-Model 

`why do heart doctors favor surgery and drugs over diet ?`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.2   | 0.1   | 0.05  | 0.04  |
| recall    | 0.026 | 0.026 | 0.026 | 0.051 |
| f1        | 0.045 | 0.041 | 0.034 | 0.045 |

R-precision: 0.05<br>
AP 0.052

-------------

`berries to prevent muscle soreness`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.4   | 0.25  | 0.12  |
| recall    | 0.057 | 0.057 | 0.071 | 0.086 |
| f1        | 0.107 | 0.1   | 0.11  | 0.1   |

R-precision: 0.1<br>
AP 0.286

-----------

`fukushima radiation and seafood`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.4   | 0.25  | 0.2   |
| recall    | 0.114 | 0.114 | 0.143 | 0.286 |
| f1        | 0.2   | 0.178 | 0.182 | 0.235 |

R-precision: 0.2<br>
AP: 0.308

-----------

`diabetes`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.9   | 0.9   | 0.82  |
| recall    | 0.007 | 0.015 | 0.035 | 0.069 |
| f1        | 0.013 | 0.03  | 0.06  | 0.128 |

R-precision: 0.302 <br>
AP: 0.685

-----------

`heart rate variability`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.2   | 0.1   | 0.05  | 0.02  |
| recall    | 0.125 | 0.125 | 0.125 | 0.125 |
| f1        | 0.154 | 0.111 | 0.071 | 0.034 |

R-precision 0.125 <br>
AP: 0.023

MAP 0.138 <br>
gemittelte 11-point AP 0.271

-----------
# LSI
`why do heart doctors favor surgery and drugs over diet ?`

| metric    | @5 | @10 | @20   | @50   |
|-----------|----|-----|-------|-------|
| precision | 0  | 0   | 0.05  | 0.04  |
| recall    | 0  | 0   | 0.026 | 0.051 |
| f1        | 0  | 0   | 0.034 | 0.045 |

R-precision: 0.0256  <br>
AP: 0.0359

-----------
`berries to prevent muscle soreness`

| metric    | @5 | @10 | @20   | @50   |
|-----------|----|-----|-------|-------|
| precision | 0  | 0   | 0.05  | 0.02  |
| recall    | 0  | 0   | 0.014 | 0.014 |
| f1        | 0  | 0   | 0.022 | 0.017 |

R-precision: 0.0286 <br>
AP: 0.023

-------------
`fukushima radiation and seafood`

| metric    | @5 | @10 | @20 | @50 |
|-----------|----|-----|-----|-----|
| precision | 0  | 0   | 0   | 0   |
| recall    | 0  | 0   | 0   | 0   |
| f1        | 0  | 0   | 0   | 0   |

R-precision: 0.0 <br>
AP: 0.014

----------------------

`diabetes`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 1     | 1     | 0.95  | 0.78  |
| recall    | 0.008 | 0.017 | 0.032 | 0.067 |
| f1        | 0.017 | 0.033 | 0.062 | 0.122 |

R-precision: 0.353 <br>
AP 0.679

-----------------
`heart rate variability`

| metric    | @5 | @10 | @20 | @50 |
|-----------|----|-----|-----|-----|
| precision | 0  | 0   | 0   | 0   |
| recall    | 0  | 0   | 0   | 0   |
| f1        | 0  | 0   | 0   | 0   |

R-precision: 0.0
AP: 0.0015

MAP 0.0032
gemittelte 11-point AP 0.1507

<img src="/images/11AP.png">
<img src="/images/precision.png">
<img src="/images/recall.png">
<img src="/images/f1.png">
