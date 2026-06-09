# Audit Probe Examples

## Sample 0

| Tick | Token | Visible | Private In | Generated Private | Action | Language | Memory | Object Pos | Pass |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | OBSERVE_OBJECT | True | 0 | 1 | RIGHT | 0 | 0/0 | 0/0 | True |
| 3 | ASK_CURRENT_POS | True | 1 | 11 | RIGHT | 17 | 0/0 | 0/0 | True |
| 8 | TOLD_GOAL | False | 1 | 1 | RIGHT | 27 | 0/0 | 0/0 | True |
| 16 | NONE | False | 5 | 17 | LEFT | 35 | 0/0 | 0/0 | True |
| 32 | NONE | False | 14 | 15 | RIGHT | 37 | 0/0 | 0/0 | True |
| 64 | NONE | False | 16 | 17 | RIGHT | 36 | 0/0 | 0/0 | True |
| 95 | NONE | False | 17 | 12 | LEFT | 39 | 0/0 | 0/0 | True |
| 111 | NONE | False | 15 | 16 | LEFT | 36 | 0/0 | 0/0 | True |

## Sample 1

| Tick | Token | Visible | Private In | Generated Private | Action | Language | Memory | Object Pos | Pass |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | OBSERVE_OBJECT | True | 0 | 3 | LEFT | 2 | 2/2 | 1/1 | True |
| 3 | ASK_CURRENT_POS | True | 3 | 11 | RIGHT | 18 | 2/2 | 1/1 | True |
| 8 | TOLD_GOAL | False | 3 | 3 | RIGHT | 29 | 2/2 | 1/1 | True |
| 16 | NONE | False | 6 | 13 | RIGHT | 35 | 2/2 | 1/1 | True |
| 32 | NONE | False | 16 | 17 | RIGHT | 39 | 2/2 | 1/1 | True |
| 64 | NONE | False | 12 | 13 | LEFT | 40 | 2/2 | 1/1 | True |
| 95 | NONE | False | 13 | 14 | RIGHT | 41 | 2/2 | 1/1 | True |
| 111 | NONE | False | 17 | 12 | STAY | 39 | 2/2 | 1/1 | True |

## Sample 2

| Tick | Token | Visible | Private In | Generated Private | Action | Language | Memory | Object Pos | Pass |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | OBSERVE_OBJECT | True | 0 | 1 | LEFT | 0 | 0/0 | 4/4 | True |
| 3 | ASK_CURRENT_POS | True | 1 | 11 | LEFT | 14 | 0/0 | 4/4 | True |
| 8 | TOLD_GOAL | False | 1 | 1 | STAY | 27 | 0/0 | 4/4 | True |
| 16 | NONE | False | 9 | 17 | STAY | 35 | 0/0 | 4/4 | True |
| 32 | NONE | False | 14 | 15 | LEFT | 37 | 0/0 | 4/4 | True |
| 64 | NONE | False | 16 | 17 | RIGHT | 36 | 0/0 | 4/4 | True |
| 95 | NONE | False | 17 | 12 | RIGHT | 41 | 0/0 | 4/4 | True |
| 111 | NONE | False | 15 | 16 | LEFT | 42 | 0/0 | 4/4 | True |
