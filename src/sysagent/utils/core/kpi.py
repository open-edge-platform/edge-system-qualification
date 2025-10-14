# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for handling KPI metrics in the core framework.
"""
from enum import Enum
from typing import Union, List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class KpiType(Enum):
    """Enumeration of supported KPI types."""
    NUMERIC = "numeric"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"


class KpiOperator(Enum):
    """Enumeration of supported KPI validation operators."""
    EQUAL = "eq"
    NOT_EQUAL = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    BETWEEN = "between"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    IN = "in"
    NOT_IN = "not_in"


class KpiSeverity(Enum):
    """Enumeration of KPI validation failure severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class KpiValidationResult:
    """Result of a KPI validation."""
    
    def __init__(self, 
                 kpi_name: str,
                 kpi_type: KpiType,
                 passed: bool,
                 actual_value: Any,
                 expected_value: Any,
                 operator: KpiOperator,
                 severity: KpiSeverity,
                 message: str = "",
                 unit: str = ""):
        """
        Initialize KPI validation result.
        
        Args:
            kpi_name: Name of the KPI
            kpi_type: Type of the KPI
            passed: Whether validation passed
            actual_value: Actual measured value
            expected_value: Expected value or threshold
            operator: Comparison operator used
            severity: Severity level if validation failed
            message: Optional descriptive message
            unit: Unit of measurement
        """
        self.kpi_name = kpi_name
        self.kpi_type = kpi_type
        self.passed = passed
        self.actual_value = actual_value
        self.expected_value = expected_value
        self.operator = operator
        self.severity = severity
        self.message = message
        self.unit = unit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'kpi_name': self.kpi_name,
            'kpi_type': self.kpi_type.value,
            'passed': self.passed,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'operator': self.operator.value if hasattr(self.operator, 'value') else str(self.operator),
            'severity': self.severity.value if hasattr(self.severity, 'value') else str(self.severity),
            'message': self.message,
            'unit': self.unit
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "PASS" if self.passed else "FAIL"
        return f"KPI {self.kpi_name}: {status} - {self.actual_value} {self.operator.value if hasattr(self.operator, 'value') else str(self.operator)} {self.expected_value}"


class KpiValidator:
    """Validator for KPI metrics."""
    
    def __init__(self):
        """Initialize KPI validator."""
        pass
    
    def validate_numeric(self, 
                        actual: Union[int, float], 
                        expected: Union[int, float, List[Union[int, float]]], 
                        operator: KpiOperator) -> bool:
        """
        Validate numeric KPI values.
        
        Args:
            actual: Actual numeric value
            expected: Expected value or range
            operator: Comparison operator
            
        Returns:
            bool: True if validation passes
        """
        try:
            if operator == KpiOperator.EQUAL:
                return actual == expected
            elif operator == KpiOperator.NOT_EQUAL:
                return actual != expected
            elif operator == KpiOperator.GREATER_THAN:
                return actual > expected
            elif operator == KpiOperator.GREATER_THAN_EQUAL:
                return actual >= expected
            elif operator == KpiOperator.LESS_THAN:
                return actual < expected
            elif operator == KpiOperator.LESS_THAN_EQUAL:
                return actual <= expected
            elif operator == KpiOperator.BETWEEN:
                if isinstance(expected, list) and len(expected) == 2:
                    return expected[0] <= actual <= expected[1]
                else:
                    raise ValueError("BETWEEN operator requires a list of two values")
            else:
                raise ValueError(f"Operator {operator} not supported for numeric values")
        except Exception as e:
            logger.error(f"Error validating numeric KPI: {e}")
            return False
    
    def validate_string(self, 
                       actual: str, 
                       expected: Union[str, List[str]], 
                       operator: KpiOperator) -> bool:
        """
        Validate string KPI values.
        
        Args:
            actual: Actual string value
            expected: Expected string or list of strings
            operator: Comparison operator
            
        Returns:
            bool: True if validation passes
        """
        try:
            if operator == KpiOperator.EQUAL:
                return actual == expected
            elif operator == KpiOperator.NOT_EQUAL:
                return actual != expected
            elif operator == KpiOperator.CONTAINS:
                return str(expected) in actual
            elif operator == KpiOperator.NOT_CONTAINS:
                return str(expected) not in actual
            elif operator == KpiOperator.MATCHES:
                return bool(re.match(str(expected), actual))
            elif operator == KpiOperator.IN:
                if isinstance(expected, list):
                    return actual in expected
                else:
                    raise ValueError("IN operator requires a list of values")
            elif operator == KpiOperator.NOT_IN:
                if isinstance(expected, list):
                    return actual not in expected
                else:
                    raise ValueError("NOT_IN operator requires a list of values")
            else:
                raise ValueError(f"Operator {operator} not supported for string values")
        except Exception as e:
            logger.error(f"Error validating string KPI: {e}")
            return False
    
    def validate_boolean(self, 
                        actual: bool, 
                        expected: bool, 
                        operator: KpiOperator) -> bool:
        """
        Validate boolean KPI values.
        
        Args:
            actual: Actual boolean value
            expected: Expected boolean value
            operator: Comparison operator
            
        Returns:
            bool: True if validation passes
        """
        try:
            if operator == KpiOperator.EQUAL:
                return actual == expected
            elif operator == KpiOperator.NOT_EQUAL:
                return actual != expected
            else:
                raise ValueError(f"Operator {operator} not supported for boolean values")
        except Exception as e:
            logger.error(f"Error validating boolean KPI: {e}")
            return False
    
    def validate_list(self, 
                     actual: List[Any], 
                     expected: Any, 
                     operator: KpiOperator) -> bool:
        """
        Validate list KPI values.
        
        Args:
            actual: Actual list value
            expected: Expected value or constraint
            operator: Comparison operator
            
        Returns:
            bool: True if validation passes
        """
        try:
            if operator == KpiOperator.CONTAINS:
                return expected in actual
            elif operator == KpiOperator.NOT_CONTAINS:
                return expected not in actual
            elif operator == KpiOperator.EQUAL:
                return actual == expected
            elif operator == KpiOperator.NOT_EQUAL:
                return actual != expected
            else:
                raise ValueError(f"Operator {operator} not supported for list values")
        except Exception as e:
            logger.error(f"Error validating list KPI: {e}")
            return False
    
    def validate_kpi(self, 
                    kpi_name: str,
                    kpi_type: KpiType,
                    actual_value: Any,
                    expected_value: Any,
                    operator: KpiOperator,
                    severity: KpiSeverity = KpiSeverity.MAJOR,
                    unit: str = "") -> KpiValidationResult:
        """
        Validate a KPI metric.
        
        Args:
            kpi_name: Name of the KPI
            kpi_type: Type of the KPI
            actual_value: Actual measured value
            expected_value: Expected value or threshold
            operator: Comparison operator
            severity: Severity level for failures
            unit: Unit of measurement
            
        Returns:
            KpiValidationResult: Validation result
        """
        try:
            if kpi_type == KpiType.NUMERIC:
                passed = self.validate_numeric(actual_value, expected_value, operator)
            elif kpi_type == KpiType.STRING:
                passed = self.validate_string(actual_value, expected_value, operator)
            elif kpi_type == KpiType.BOOLEAN:
                passed = self.validate_boolean(actual_value, expected_value, operator)
            elif kpi_type == KpiType.LIST:
                passed = self.validate_list(actual_value, expected_value, operator)
            else:
                raise ValueError(f"Unknown KPI type: {kpi_type}")
            
            message = f"KPI validation {'passed' if passed else 'failed'}"
            
            return KpiValidationResult(
                kpi_name=kpi_name,
                kpi_type=kpi_type,
                passed=passed,
                actual_value=actual_value,
                expected_value=expected_value,
                operator=operator,
                severity=severity,
                message=message,
                unit=unit
            )
            
        except Exception as e:
            logger.error(f"Error validating KPI {kpi_name}: {e}")
            return KpiValidationResult(
                kpi_name=kpi_name,
                kpi_type=kpi_type,
                passed=False,
                actual_value=actual_value,
                expected_value=expected_value,
                operator=operator,
                severity=KpiSeverity.CRITICAL,
                message=f"Validation error: {str(e)}",
                unit=unit
            )


def validate_kpi_batch(kpi_definitions: List[Dict[str, Any]], 
                      actual_values: Dict[str, Any]) -> List[KpiValidationResult]:
    """
    Validate multiple KPIs in batch.
    
    Args:
        kpi_definitions: List of KPI definition dictionaries
        actual_values: Dictionary of actual measured values
        
    Returns:
        List of KPI validation results
    """
    validator = KpiValidator()
    results = []
    
    for kpi_def in kpi_definitions:
        try:
            kpi_name = kpi_def['name']
            kpi_type = KpiType(kpi_def['type'])
            expected_value = kpi_def['expected']
            operator = KpiOperator(kpi_def['operator'])
            severity = KpiSeverity(kpi_def.get('severity', 'major'))
            unit = kpi_def.get('unit', '')
            
            actual_value = actual_values.get(kpi_name)
            if actual_value is None:
                logger.warning(f"No actual value found for KPI: {kpi_name}")
                continue
            
            result = validator.validate_kpi(
                kpi_name=kpi_name,
                kpi_type=kpi_type,
                actual_value=actual_value,
                expected_value=expected_value,
                operator=operator,
                severity=severity,
                unit=unit
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing KPI definition: {e}")
    
    return results


# Standalone functions for backward compatibility
def validate_numeric_kpi(value: Union[int, float], 
                         kpi_config: Dict[str, Any]) -> KpiValidationResult:
    """
    Validate a numeric KPI value against its configuration.
    
    Args:
        value: The actual numeric value to validate
        kpi_config: The KPI configuration dictionary
    
    Returns:
        KpiValidationResult: The result of the validation
    """
    validation = kpi_config.get("validation", {})
    if not validation.get("enabled", True):
        # If validation is disabled, always return passed
        return KpiValidationResult(
            kpi_name=kpi_config.get("name", "Unnamed KPI"),
            kpi_type=KpiType.NUMERIC,
            passed=True,
            actual_value=value,
            expected_value=validation.get("reference", None),
            operator=validation.get("operator", None),
            severity=kpi_config.get("severity", KpiSeverity.INFO),
            message=kpi_config.get("description", ""),
            unit=kpi_config.get("unit", "")
        )
    
    operator = validation.get("operator")
    reference = validation.get("reference")
    passed = False
    
    if operator == KpiOperator.EQUAL.value:
        passed = value == reference
    elif operator == KpiOperator.NOT_EQUAL.value:
        passed = value != reference
    elif operator == KpiOperator.GREATER_THAN.value:
        passed = value > reference
    elif operator == KpiOperator.GREATER_THAN_EQUAL.value:
        passed = value >= reference
    elif operator == KpiOperator.LESS_THAN.value:
        passed = value < reference
    elif operator == KpiOperator.LESS_THAN_EQUAL.value:
        passed = value <= reference
    elif operator == KpiOperator.BETWEEN.value and isinstance(reference, list) and len(reference) == 2:
        passed = reference[0] <= value <= reference[1]
    
    return KpiValidationResult(
        kpi_name=kpi_config.get("name", "Unnamed KPI"),
        kpi_type=KpiType.NUMERIC,
        passed=passed,
        actual_value=value,
        expected_value=reference,
        operator=operator,
        severity=kpi_config.get("severity", KpiSeverity.INFO),
        message=kpi_config.get("description", ""),
        unit=kpi_config.get("unit", "")
    )


def validate_string_kpi(value: str, 
                       kpi_config: Dict[str, Any]) -> KpiValidationResult:
    """
    Validate a string KPI value against its configuration.
    
    Args:
        value: The actual string value to validate
        kpi_config: The KPI configuration dictionary
    
    Returns:
        KpiValidationResult: The result of the validation
    """
    validation = kpi_config.get("validation", {})
    if not validation.get("enabled", True):
        # If validation is disabled, always return passed
        return KpiValidationResult(
            kpi_name=kpi_config.get("name", "Unnamed KPI"),
            kpi_type=KpiType.STRING,
            passed=True,
            actual_value=value,
            expected_value=validation.get("reference", None),
            operator=validation.get("operator", None),
            severity=kpi_config.get("severity", KpiSeverity.INFO),
            message=kpi_config.get("description", ""),
            unit=kpi_config.get("unit", "")
        )
    
    operator = validation.get("operator")
    reference = validation.get("reference")
    passed = False
    
    if operator == KpiOperator.EQUAL.value:
        passed = value == reference
    elif operator == KpiOperator.NOT_EQUAL.value:
        passed = value != reference
    elif operator == KpiOperator.CONTAINS.value:
        passed = reference in value
    elif operator == KpiOperator.NOT_CONTAINS.value:
        passed = reference not in value
    elif operator == KpiOperator.MATCHES.value:
        import re
        passed = bool(re.match(reference, value))
    elif operator == KpiOperator.IN.value and isinstance(reference, list):
        passed = value in reference
    elif operator == KpiOperator.NOT_IN.value and isinstance(reference, list):
        passed = value not in reference
    
    return KpiValidationResult(
        kpi_name=kpi_config.get("name", "Unnamed KPI"),
        kpi_type=KpiType.STRING,
        passed=passed,
        actual_value=value,
        expected_value=reference,
        operator=operator,
        severity=kpi_config.get("severity", KpiSeverity.INFO),
        message=kpi_config.get("description", ""),
        unit=kpi_config.get("unit", "")
    )


def validate_boolean_kpi(value: bool, 
                         kpi_config: Dict[str, Any]) -> KpiValidationResult:
    """
    Validate a boolean KPI value against its configuration.
    
    Args:
        value: The actual boolean value to validate
        kpi_config: The KPI configuration dictionary
    
    Returns:
        KpiValidationResult: The result of the validation
    """
    validation = kpi_config.get("validation", {})
    if not validation.get("enabled", True):
        # If validation is disabled, always return passed
        return KpiValidationResult(
            kpi_name=kpi_config.get("name", "Unnamed KPI"),
            kpi_type=KpiType.BOOLEAN,
            passed=True,
            actual_value=value,
            expected_value=validation.get("reference", None),
            operator=validation.get("operator", None),
            severity=kpi_config.get("severity", KpiSeverity.INFO),
            message=kpi_config.get("description", ""),
            unit=kpi_config.get("unit", "")
        )
    
    reference = validation.get("reference")
    passed = value == reference
    
    return KpiValidationResult(
        kpi_name=kpi_config.get("name", "Unnamed KPI"),
        kpi_type=KpiType.BOOLEAN,
        passed=passed,
        actual_value=value,
        expected_value=reference,
        operator=KpiOperator.EQUAL.value,
        severity=kpi_config.get("severity", KpiSeverity.INFO),
        message=kpi_config.get("description", ""),
        unit=kpi_config.get("unit", "")
    )


def validate_list_kpi(value: List[Any], 
                     kpi_config: Dict[str, Any]) -> KpiValidationResult:
    """
    Validate a list KPI value against its configuration.
    
    Args:
        value: The actual list value to validate
        kpi_config: The KPI configuration dictionary
    
    Returns:
        KpiValidationResult: The result of the validation
    """
    validation = kpi_config.get("validation", {})
    if not validation.get("enabled", True):
        # If validation is disabled, always return passed
        return KpiValidationResult(
            kpi_name=kpi_config.get("name", "Unnamed KPI"),
            kpi_type=KpiType.LIST,
            passed=True,
            actual_value=value,
            expected_value=validation.get("reference", None),
            operator=validation.get("operator", None),
            severity=kpi_config.get("severity", KpiSeverity.INFO),
            message=kpi_config.get("description", ""),
            unit=kpi_config.get("unit", "")
        )
    
    operator = validation.get("operator")
    reference = validation.get("reference")
    passed = False
    
    if operator == KpiOperator.EQUAL.value:
        passed = value == reference
    elif operator == KpiOperator.CONTAINS.value:
        passed = all(item in value for item in reference)
    elif operator == KpiOperator.NOT_CONTAINS.value:
        passed = all(item not in value for item in reference)
    elif operator == KpiOperator.IN.value:
        passed = all(item in reference for item in value)
    elif operator == KpiOperator.NOT_IN.value:
        passed = all(item not in reference for item in value)
    
    return KpiValidationResult(
        kpi_name=kpi_config.get("name", "Unnamed KPI"),
        kpi_type=KpiType.LIST,
        passed=passed,
        actual_value=value,
        expected_value=reference,
        operator=operator,
        severity=kpi_config.get("severity", KpiSeverity.INFO),
        message=kpi_config.get("description", ""),
        unit=kpi_config.get("unit", "")
    )


def validate_kpi(value: Any, kpi_config: Dict[str, Any]) -> KpiValidationResult:
    """
    Validate a KPI value against its configuration.
    
    Args:
        value: The actual value to validate
        kpi_config: The KPI configuration dictionary
    
    Returns:
        KpiValidationResult: The result of the validation
    """
    kpi_type = kpi_config.get("type")
    
    if kpi_type == KpiType.NUMERIC.value:
        return validate_numeric_kpi(value, kpi_config)
    elif kpi_type == KpiType.STRING.value:
        return validate_string_kpi(value, kpi_config)
    elif kpi_type == KpiType.BOOLEAN.value:
        return validate_boolean_kpi(value, kpi_config)
    elif kpi_type == KpiType.LIST.value:
        return validate_list_kpi(value, kpi_config)
    else:
        raise ValueError(f"Unsupported KPI type: {kpi_type}")
