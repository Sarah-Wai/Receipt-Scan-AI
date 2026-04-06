from typing import List, Dict, Any, Optional, Tuple
import statistics


class ItemValidationEngine:
    """
    Multi-level validation for reconstructed receipt items.

    Validates items against:
    1. Subtotal consistency
    2. Price range expectations
    3. Item name quality (OCR errors)
    4. Outlier detection
    5. Label sequence consistency
    """

    def __init__(self, subtotal: Optional[float] = None, tax: Optional[float] = None):
        self.subtotal = subtotal
        self.tax = tax
        self.validation_rules = []

    def _safe_price(self, value: Any) -> Optional[float]:
        """Convert raw price into float, else return None."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def validate_all(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "subtotal_check": self.check_subtotal_consistency(items),
            "price_outliers": self.detect_price_outliers(items),
            "name_quality": self.check_item_name_quality(items),
            "price_range": self.check_price_ranges(items),
            "label_sequence": self.check_label_sequence_quality(items),
            "recommendations": self.get_recommendations(items),
        }

    # ========================================================================
    # 1) SUBTOTAL CONSISTENCY CHECK
    # ========================================================================

    def check_subtotal_consistency(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.subtotal is None:
            return {"status": "SKIPPED", "reason": "No subtotal provided"}

        valid_prices = []
        for item in items:
            price = self._safe_price(item.get("price"))
            if price is not None and price > 0:
                valid_prices.append(price)

        reconstructed_total = sum(valid_prices)
        discrepancy = reconstructed_total - self.subtotal
        discrepancy_pct = (discrepancy / self.subtotal * 100) if self.subtotal else 0

        if abs(discrepancy_pct) < 1:
            status = "PASS"
        elif abs(discrepancy_pct) < 5:
            status = "WARNING"
        else:
            status = "FAIL"

        problematic = self.find_problematic_items(items, discrepancy)

        return {
            "status": status,
            "reconstructed_total": round(reconstructed_total, 2),
            "expected_subtotal": self.subtotal,
            "discrepancy": round(discrepancy, 2),
            "discrepancy_pct": round(discrepancy_pct, 2),
            "problematic_items": problematic,
            "message": self._subtotal_message(status, discrepancy, discrepancy_pct),
        }

    def find_problematic_items(
        self, items: List[Dict[str, Any]], discrepancy: float
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        prices = []
        for item in items:
            price = self._safe_price(item.get("price"))
            if price is not None and price > 0:
                prices.append(price)

        if not prices:
            return []

        avg_price = statistics.mean(prices)
        std_price = statistics.stdev(prices) if len(prices) > 1 else 0

        problematic = []
        for item in items:
            price = self._safe_price(item.get("price"))
            if price is None or price <= 0:
                continue

            if std_price > 0 and abs(price - avg_price) > 2 * std_price:
                z_score = (price - avg_price) / std_price
                problematic.append({
                    "item_name": item.get("item_name"),
                    "price": price,
                    "avg_price": round(avg_price, 2),
                    "z_score": round(z_score, 2),
                    "likelihood": "HIGH" if abs(z_score) > 3 else "MEDIUM",
                })

        return problematic

    def _subtotal_message(self, status: str, discrepancy: float, pct: float) -> str:
        if status == "PASS":
            return f"✓ Subtotal matches (discrepancy: {pct:.2f}%)"
        elif status == "WARNING":
            return f"⚠ Slight discrepancy: ${discrepancy:+.2f} ({pct:+.2f}%)"
        else:
            return f"✗ Large discrepancy: ${discrepancy:+.2f} ({pct:+.2f}%) — likely OCR/labeling errors"

    # ========================================================================
    # 2) PRICE OUTLIER DETECTION
    # ========================================================================

    def detect_price_outliers(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        prices = []
        for item in items:
            price = self._safe_price(item.get("price"))
            if price is not None and price > 0:
                prices.append(price)

        if len(prices) < 3:
            return {"status": "INSUFFICIENT_DATA", "outliers": []}

        mean_price = statistics.mean(prices)
        std_price = statistics.stdev(prices) if len(prices) > 1 else 0
        median_price = statistics.median(prices)

        sorted_prices = sorted(prices)
        q1 = sorted_prices[len(sorted_prices) // 4]
        q3 = sorted_prices[3 * len(sorted_prices) // 4]
        iqr = q3 - q1

        outliers = []
        for item in items:
            price = self._safe_price(item.get("price"))
            if price is None or price <= 0:
                continue

            z_score = (price - mean_price) / std_price if std_price > 0 else 0
            is_outlier_iqr = (price < q1 - 1.5 * iqr) or (price > q3 + 1.5 * iqr)

            if abs(z_score) > 2.5 or is_outlier_iqr:
                outliers.append({
                    "item_name": item.get("item_name"),
                    "price": price,
                    "z_score": round(z_score, 2),
                    "is_outlier_iqr": is_outlier_iqr,
                    "reason": self._outlier_reason(z_score, is_outlier_iqr, mean_price, median_price),
                })

        return {
            "status": "OUTLIERS_FOUND" if outliers else "OK",
            "mean_price": round(mean_price, 2),
            "median_price": round(median_price, 2),
            "std_dev": round(std_price, 2),
            "outliers": outliers,
        }

    def _outlier_reason(self, z_score: float, is_iqr: bool, mean: float, median: float) -> str:
        if z_score > 2.5:
            return f"Z-score {z_score:.1f} (>> mean {mean:.2f})"
        elif z_score < -2.5:
            return f"Z-score {z_score:.1f} (<< mean {mean:.2f})"
        elif is_iqr:
            return "Beyond IQR bounds"
        return "Outlier detected"

    # ========================================================================
    # 3) ITEM NAME QUALITY CHECK
    # ========================================================================

    def check_item_name_quality(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        issues = []

        for item in items:
            name = str(item.get("item_name", "") or "")
            conf = item.get("conf_mean", 1.0)

            try:
                conf = float(conf) if conf is not None else 1.0
            except (TypeError, ValueError):
                conf = 1.0

            quality_issues = []

            if self._has_ocr_artifacts(name):
                quality_issues.append("OCR artifacts (0/O, 1/I, 5/S confusion)")

            digit_ratio = sum(1 for c in name if c.isdigit()) / len(name) if name else 0
            if digit_ratio > 0.4:
                quality_issues.append(f"High digit ratio ({digit_ratio:.1%}) — likely SKU/barcode mixed")

            if len(name) < 3:
                quality_issues.append("Name too short — likely SKU only")
            elif len(name) > 40:
                quality_issues.append("Name too long — likely multiple items merged")

            if conf < 0.65:
                quality_issues.append(f"Low confidence ({conf:.2f})")

            if quality_issues:
                issues.append({
                    "item_name": name,
                    "price": item.get("price"),
                    "conf": conf,
                    "issues": quality_issues,
                })

        return {
            "status": "ISSUES_FOUND" if issues else "OK",
            "num_issues": len(issues),
            "issues": issues[:10],
        }

    def _has_ocr_artifacts(self, text: str) -> bool:
        patterns = {
            "0O": "0 and O confused",
            "1l": "1 and l confused",
            "5S": "5 and S confused",
            "8B": "8 and B confused",
            "rn": "rn appears as m",
        }

        for pattern in patterns:
            if pattern.lower() in text.lower():
                return True

        return False

    # ========================================================================
    # 4) PRICE RANGE CHECK
    # ========================================================================

    def check_price_ranges(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        suspicious = []

        for item in items:
            price = self._safe_price(item.get("price"))

            if price is None:
                continue
            elif price <= 0:
                suspicious.append((item, "Price <= 0 (likely OCR error)"))
            elif price < 0.50:
                suspicious.append((item, f"Very low price: ${price:.2f}"))
            elif price > 500:
                suspicious.append((item, f"Suspiciously high: ${price:.2f}"))
            elif price > 100 and "THHCFRENCH" in str(item.get("item_name", "")).upper():
                suspicious.append((item, f"Item '{item.get('item_name')}' too expensive: ${price:.2f}"))

        return {
            "status": "WARNINGS" if suspicious else "OK",
            "suspicious_prices": [
                {
                    "item_name": item.get("item_name"),
                    "price": item.get("price"),
                    "reason": reason,
                }
                for item, reason in suspicious
            ],
        }

    # ========================================================================
    # 5) LABEL SEQUENCE QUALITY
    # ========================================================================

    def check_label_sequence_quality(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        issues = []

        for item in items:
            tokens = item.get("tokens", [])
            if not tokens:
                continue

            first_token = str(tokens[0])
            if not (self._is_sku(first_token) or self._looks_like_item_name(first_token)):
                issues.append({
                    "item_name": item.get("item_name"),
                    "first_token": first_token,
                    "issue": "Does not start with SKU or item name",
                })

        return {
            "status": "ISSUES" if issues else "OK",
            "issues": issues,
        }

    def _is_sku(self, token: str) -> bool:
        return len(token) >= 6 and token.isdigit()

    def _looks_like_item_name(self, token: str) -> bool:
        return len(token) >= 2 and not token.isdigit()

    # ========================================================================
    # 6) RECOMMENDATIONS
    # ========================================================================

    def get_recommendations(self, items: List[Dict[str, Any]]) -> List[str]:
        recommendations = []

        subtotal_check = self.check_subtotal_consistency(items)
        if subtotal_check["status"] == "FAIL":
            problematic = subtotal_check.get("problematic_items", [])
            if problematic:
                names = [p["item_name"] for p in problematic[:3]]
                recommendations.append(
                    f"✗ Subtotal mismatch: Review items {names} (likely OCR errors or label misalignment)"
                )

        outliers = self.detect_price_outliers(items)
        if outliers["status"] == "OUTLIERS_FOUND":
            outlier_names = [o["item_name"] for o in outliers["outliers"][:2]]
            recommendations.append(
                f"⚠ Price outliers detected: {outlier_names}. Verify these prices."
            )

        name_quality = self.check_item_name_quality(items)
        if name_quality["status"] == "ISSUES_FOUND":
            recommendations.append(
                f"⚠ {name_quality['num_issues']} items have OCR quality issues. Manual review recommended."
            )

        price_ranges = self.check_price_ranges(items)
        if price_ranges["status"] == "WARNINGS":
            recommendations.append(
                f"⚠ {len(price_ranges['suspicious_prices'])} items have suspicious prices."
            )

        if not recommendations:
            recommendations.append("✓ All validations passed. Receipt looks good!")

        return recommendations