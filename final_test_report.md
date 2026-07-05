# Final Test Report

This report summarizes the generated test cases and the comparison results between the R and Python implementations.

## Summary

| File / Function | R Testcases | Python Testcases | Matched / Passed | Differences / Failed | Result |
|---|---:|---:|---:|---:|:---:|
| `randomDAG` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `allDagsIntern` | 1,500 | 1,500 | 1,500 | 0 | Passed |
| `allDagsJonas` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `computeCausOrder` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `computePathMatrix` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `computePathMatrix2` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `dag2cpdagAdj` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `dSepAdji` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `hammingDist` | 10,000 | 10,000 | 10,000 | 0 | Passed |
| `structIntervDist` | 10,000 | 10,000 | 10,000 | 0 | Passed |

## Detailed Results

### randomDAG

For `randomDAG`, 10000 R test cases and 10000 Python test cases were generated and checked. All 10000 test cases passed the property-based checks, with 0 failed test cases.

### allDagsIntern

For `allDagsIntern`, 1500 R test cases and 1500 Python test cases were generated and compared. All 1500 test cases matched, and 0 differences were found.

A larger batch of 10000 test cases was not used for this file because `allDagsIntern` recursively enumerates valid DAG expansions. For some graph structures, the number of possible expansions grows very quickly, which made larger batches extremely slow and caused the execution to get stuck on heavier cases. Therefore, this file was tested with 1500 cases, and all results matched successfully.

### allDagsJonas

For `allDagsJonas`, 10000 R test cases and 10000 Python test cases were generated and compared. All 10000 test cases matched, and 0 differences were found.

### computeCausOrder

For `computeCausOrder`, 10000 R test cases and 10000 Python test cases were generated and compared. No differences were found, and all testcase matrices and results were identical.

### computePathMatrix

For `computePathMatrix`, 10000 R test cases and 10000 Python test cases were generated and compared. No differences were found, and all testcase matrices and result matrices were identical.

### computePathMatrix2

For `computePathMatrix2`, 10000 R test cases and 10000 Python test cases were generated and compared. All 10000 test cases matched, and 0 differences were found.

### dag2cpdagAdj

For `dag2cpdagAdj`, 10000 R test cases and 10000 Python test cases were generated and compared. All 10000 test cases matched, and 0 differences were found.

### dSepAdji

For `dSepAdji`, 10000 R test cases and 10000 Python test cases were generated and compared. All 10000 test cases matched, and 0 differences were found.

### hammingDist

For `hammingDist`, 10000 R test cases and 10000 Python test cases were generated and compared. All 10000 test cases matched, and 0 differences were found.

### structIntervDist

For `structIntervDist`, 10000 R test cases and 10000 Python test cases were generated and compared using a SID-specific comparison script. All 10000 test cases matched, and 0 differences were found.

The comparison checked the main SID outputs, including `sid`, `sidUpperBound`, `sidLowerBound`, and `incorrectMat`. The testcase inputs and SID results were identical for all compared cases.

## Additional Validation

In addition to the automatic comparison, I also manually introduced a few incorrect outputs to verify that the comparison scripts were able to detect mismatches. The scripts correctly reported the differences, which confirms that the comparison checks are working as expected.

## Files Included in the Repository

For each tested function, the repository contains the generated test cases, the R outputs, the Python outputs, the comparison script, and the comparison report.

## Conclusion

All tested R and Python implementations produced matching outputs for the generated test cases. No differences were found in the comparison reports.
