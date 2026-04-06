"""
Item Flag Enricher - Adds review flags based on validation results
Works AFTER ItemValidationEngine to identify items needing manual review
"""

from typing import List, Dict, Any
from enum import Enum


class ReviewFlag(str, Enum):
    """Flag types for human-in-the-loop review"""
    CLEAN = "clean"
    PRICE_OUTLIER = "price_outlier"
    MISSING_PRICE = "missing_price"


class ItemFlagEnricher:
    """
    Adds review flags based on ItemValidationEngine validation results.
    
    Only flags items with ACTUAL DATA ISSUES:
    - MISSING_PRICE: price is None
    - PRICE_OUTLIER: detected by validation engine
    - CLEAN: no validation issues
    
    Does NOT flag based on confidence - only on validation results.
    
    Usage in notebook:
    ```
    # Step 1: Run validation
    validator = ItemValidationEngine(subtotal=extracted_subtotal)
    validation_report = validator.validate_all(item_lines)
    
    # Step 2: Add flags based on validation
    enricher = ItemFlagEnricher()
    item_lines_with_flags = enricher.add_flags_from_validation(
        item_lines, 
        validation_report
    )
    
    # Separate items
    clean_items = enricher.get_clean_items(item_lines_with_flags)
    flagged_only = enricher.get_flagged_items(item_lines_with_flags)
    ```
    """
    
    def __init__(self):
        """Initialize enricher"""
        pass
    
    def add_flags_from_validation(
        self, 
        item_lines: List[Dict[str, Any]], 
        validation_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add flags to items based on validation report.
        
        Args:
            item_lines: List of items from reconstruction
            validation_report: Output from ItemValidationEngine.validate_all()
        
        Returns:
            Items with added 'flag' and 'flag_reason' fields
        """
        # Extract outlier item names from validation report
        outlier_names = set()
        
        print(f"DEBUG: Validation report keys: {validation_report.keys()}")
        
        # Method 1: Check price_range (contains suspicious_prices)
        if "price_range" in validation_report:
            price_range_report = validation_report["price_range"]
            print(f"DEBUG: price_range keys: {price_range_report.keys()}")
            suspicious = price_range_report.get("suspicious_prices", [])
            for item_info in suspicious:
                name = item_info.get("item_name")
                print(f"DEBUG: Found outlier from price_range: {name}")
                outlier_names.add(name)
        
        # Method 2: Check price_outliers (another possible field)
        if "price_outliers" in validation_report:
            price_outliers = validation_report["price_outliers"]
            print(f"DEBUG: price_outliers keys: {price_outliers.keys()}")
            suspicious = price_outliers.get("suspicious_prices", [])
            for item_info in suspicious:
                name = item_info.get("item_name")
                print(f"DEBUG: Found outlier from price_outliers: {name}")
                outlier_names.add(name)
        
        # Method 3: Check detect_price_outliers
        if "detect_price_outliers" in validation_report:
            detected = validation_report["detect_price_outliers"]
            print(f"DEBUG: detect_price_outliers keys: {detected.keys()}")
            outliers = detected.get("outliers", [])
            for item_info in outliers:
                name = item_info.get("item_name")
                print(f"DEBUG: Found outlier from detect_price_outliers: {name}")
                outlier_names.add(name)
        
        print(f"DEBUG: Total outliers found: {outlier_names}")
        
        # Add flags to items
        result = []
        for item in item_lines:
            item = item.copy()
            item_name = item.get("item_name", "")
            price = item.get("price")
            
            # Determine flag
            if price is None:
                item["flag"] = ReviewFlag.MISSING_PRICE.value
                item["flag_reason"] = "Missing price - not extracted from receipt"
            elif item_name in outlier_names:
                item["flag"] = ReviewFlag.PRICE_OUTLIER.value
                item["flag_reason"] = f"Price outlier detected: ${price:.2f}"
                print(f"DEBUG: Flagging {item_name} as PRICE_OUTLIER")
            else:
                item["flag"] = ReviewFlag.CLEAN.value
                item["flag_reason"] = "OK - passed validation"
            
            result.append(item)
        
        return result
    
    def get_clean_items(self, flagged_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter items with CLEAN flag"""
        return [item for item in flagged_items if item.get("flag") == ReviewFlag.CLEAN.value]
    
    def get_flagged_items(self, flagged_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter items with any flag (not CLEAN)"""
        return [item for item in flagged_items if item.get("flag") != ReviewFlag.CLEAN.value]
    
    def get_clean(self, flagged_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alias for get_clean_items"""
        return self.get_clean_items(flagged_items)
    
    def get_flagged(self, flagged_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alias for get_flagged_items"""
        return self.get_flagged_items(flagged_items)
    
    def get_items_by_flag(self, flagged_items: List[Dict[str, Any]], flag: str) -> List[Dict[str, Any]]:
        """Filter items by specific flag"""
        return [item for item in flagged_items if item.get("flag") == flag]
    
    def summary(self, flagged_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of flags"""
        if not flagged_items:
            return {
                "total": 0,
                "clean": 0,
                "flagged": 0,
                "clean_pct": "0%",
                "breakdown": {},
            }
        
        breakdown = {}
        for flag in ReviewFlag:
            count = len(self.get_items_by_flag(flagged_items, flag.value))
            if count > 0:
                breakdown[flag.value] = count
        
        clean_count = len(self.get_clean_items(flagged_items))
        flagged_count = len(flagged_items) - clean_count
        clean_pct = f"{clean_count / len(flagged_items) * 100:.1f}%" if flagged_items else "0%"
        
        return {
            "total": len(flagged_items),
            "clean": clean_count,
            "flagged": flagged_count,
            "clean_pct": clean_pct,
            "breakdown": breakdown,
        }