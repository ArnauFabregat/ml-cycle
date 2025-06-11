# Unit tests

## Structure
- Folder organization like the library
- A single .py with tests for each .py of the library
- Single class with tests for each class in the library
- Common test class for every function inside each .py
- Each .py in the library must have:
    - Tests for typing input format
    - Tests for input accepted values
    - Functionality tests


## Parameter initialization:
- If a parameter is used in 2 or more unit tests: initiate inside **setUp** `@classmethod`

```python
import unittest


class TestExampleClass(unittest.TestCase):
  @classmethod
  def setUp(self) -> None:
    self.param_1 = value_1
    self.param_2 = value_2
      
  def test_ExampleTest_1_TypeError(self):
    self.assertRaises(TypeError,
                      method_to_test_1,
                      method_1_param_1=self.param_1,
                      method_2_param_2=self.param_2)
      
  def test_ExampleTest_2_TypeError(self):
    self.assertRaises(TypeError,
                      method_to_test_2,
                      method_2_param_1=self.param_1,
                      method_2_param_2=self.param_2)
```
- If a parameter is used just in one unit test: initiate inside unit test method

```python
import unittest


class TestExampleClass(unittest.TestCase):
  def test_ExampleTest_TypeError(self):
    param_1 = value_1
    param_2 = value_2
    self.assertRaises(TypeError,
                      method_to_test,
                      method_param_1=param_1,
                      method_param_2=param_2)
``` 
- If a parameter is used globally: initiate at the beginning of the .py

```python
import unittest

global_param_1 = value_1
global_param_2 = value_2

class TestExampleClass_1(unittest.TestCase):
  def test_ExampleTest_1_TypeError(self):
    self.assertRaises(TypeError,
                      method_to_test_1,
                      method_param_1=global_param_1,
                      method_param_2=global_param_2)

class TestExampleClass_2(unittest.TestCase):
  def test_ExampleTest_2_TypeError(self):
    self.assertRaises(TypeError,
                      method_to_test_2,
                      method_param_1=global_param_1,
                      method_param_2=global_param_2)
```

## Naming convention
- file.py: `test_{filename}.py`
    - *Example*: repclass/pricing/hard_optimizers/probability_models/monotonic_LGBM.py  ->  **test**/pricing/hard_optimizers/probability_models/**test**_monotonic_LGBM.py

- class/function: `TestClassName`/`TestFunctionName`
    - *Example*: `repclass.pricing.hard_optimizers.probability_models.monotonic_LGBM.monotonic_LGBM`  ->  `class TestMonotonicLGBM(unittest.TestCase)`
    - *Example*: `repclass.etl.etl_functions.ci__mean_upper_bound`  ->  `class TestCiMeanUpperBound(unittest.TestCase)`

- Test methods (function): `test_StateUnderTest_ExpectedBehaviour`
    - *Example*: `ci_mean_upper_bound()`  ->  

```python
# repclass/etl/test_etl_functions.py

import unittest
from src.repclass.etl.etl_functions import ci_mean_upper_bound


class TestCiMeanUpperBound(unittest.TestCase):
    def test_IncorrectInputFormat_TypeError(self):
        self.assertRaises(TypeError,
                          ci_mean_upper_bound,
                          *args,
                          **kwds)

    def test_OutputEqual_True(self):
        self.assertEqual(...)
```

- Test methods (class): `test_MethodName_StateUnderTest_ExpectedBehaviour`
    - *Example*: `TransaccionalInputTable.init()` y `TransaccionalInputTable._agg_feat()`  ->  

```python
# repclass/etl/test_transaccional.py

import unittest
from src.repclass.etl.transaccional import TransaccionalInputTable


class TestTransaccionalInputTable(unittest.TestCase):
    def test_Init_IncorrectInputFormat_TypeError(self):
        self.assertRaises(TypeError,
                          TransaccionalInputTable,
                          *args,
                          **kwds)

    def test_AggFeat_OutputDataFrameEqual_True(self):
      self.assertEqual(...)
```
