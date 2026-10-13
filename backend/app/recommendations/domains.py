"""Domain detection for recommendation catalogs.

Supports multi-domain intelligence: Telecom, HR Attrition, Finance/Banking,
Healthcare, E-Commerce, SaaS, and Generic business prediction tasks.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

DOMAIN_TELCO = "telco"
DOMAIN_HR_ATTRITION = "hr_attrition"
DOMAIN_FINANCE = "finance"
DOMAIN_HEALTHCARE = "healthcare"
DOMAIN_ECOMMERCE = "ecommerce"
DOMAIN_SAAS = "saas"
DOMAIN_GENERIC = "generic"

DEFAULT_DOMAIN = DOMAIN_TELCO

DOMAIN_DISPLAY_NAMES = {
    DOMAIN_TELCO: "Telecom & Subscriptions",
    DOMAIN_HR_ATTRITION: "HR & Workforce Retention",
    DOMAIN_FINANCE: "Banking & Financial Risk",
    DOMAIN_HEALTHCARE: "Healthcare & Patient Care",
    DOMAIN_ECOMMERCE: "E-Commerce & Retail",
    DOMAIN_SAAS: "B2B & SaaS Products",
    DOMAIN_GENERIC: "Enterprise Operations (General)",
}

DOMAIN_UNIT_LABELS = {
    DOMAIN_TELCO: "Saved Subscriber",
    DOMAIN_HR_ATTRITION: "Retained Employee",
    DOMAIN_FINANCE: "Protected Account / Loan",
    DOMAIN_HEALTHCARE: "Prevented Readmission",
    DOMAIN_ECOMMERCE: "Retained Shopper",
    DOMAIN_SAAS: "Retained Account",
    DOMAIN_GENERIC: "Saved Case / Account",
}

_HR_FEATURE_MARKERS = {
    "overtime",
    "jobsatisfaction",
    "environmentsatisfaction",
    "worklifebalance",
    "yearsatcompany",
    "yearssincelastpromotion",
    "yearswithcurrmanager",
    "distancefromhome",
    "businesstravel",
    "monthlyincome",
    "numcompaniesworked",
    "stockoptionlevel",
    "trainingtimeslastyear",
    "jobinvolvement",
    "relationshipsatisfaction",
    "totalworkingyears",
    "percentsalaryhike",
}

_TELCO_FEATURE_MARKERS = {
    "monthlycharges",
    "totalcharges",
    "contract",
    "techsupport",
    "onlinesecurity",
    "internetservice",
    "streamingtv",
    "paymentmethod",
    "tenure",
    "phoneservice",
}

_FINANCE_FEATURE_MARKERS = {
    "creditscore",
    "loanamount",
    "interestrate",
    "debttoincome",
    "annualincome",
    "revolvingbalance",
    "delinquent",
    "fico",
    "installment",
    "openacc",
    "revolutil",
    "mortgage",
    "creditlimit",
    "bankruptcy",
}

_HEALTHCARE_FEATURE_MARKERS = {
    "nummedications",
    "numlabprocedures",
    "timeinhospital",
    "admissiontype",
    "diagnoses",
    "glucose",
    "insulin",
    "readmitted",
    "hospital",
    "patient",
    "dosage",
    "clinical",
}

_ECOMMERCE_FEATURE_MARKERS = {
    "cartvalue",
    "sessionduration",
    "pageviews",
    "itemsincart",
    "totalorders",
    "purchases",
    "discountused",
    "dayssincelastpurchase",
    "bounces",
    "addtocart",
}

_SAAS_FEATURE_MARKERS = {
    "seatsused",
    "activeusers",
    "dailylogins",
    "apicalls",
    "supporttickets",
    "featureadoption",
    "mrr",
    "arr",
    "seatutilization",
    "license",
}


def _norm_token(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    s = s.lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "", s).strip("_")  # collapse for marker match


def detect_domain(
    organization_industry: Optional[str] = None,
    feature_columns: Optional[Iterable[str]] = None,
    features: Optional[dict[str, Any]] = None,
    project_name: Optional[str] = None,
    target_column: Optional[str] = None,
    target_description: Optional[str] = None,
    metadata_text: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Infer recommendation domain from organization industry, project metadata, and features.

    Preference order:
    1. Direct organization industry match (if set)
    2. Explicit keywords in project name, target column, or description
    3. Statistical column feature-set markers
    4. Fallback to generic or telco if ambiguous.
    """
    # 1. Organization industry direct check
    ind = str(organization_industry or "").strip().lower()
    if any(k in ind for k in ("finance", "banking", "lending", "credit")):
        return DOMAIN_FINANCE
    elif any(k in ind for k in ("healthcare", "health", "hospital", "pharma", "medical", "clinic")):
        return DOMAIN_HEALTHCARE
    elif any(k in ind for k in ("ecommerce", "e-commerce", "retail", "store", "shopping")):
        return DOMAIN_ECOMMERCE
    elif any(k in ind for k in ("saas", "software", "tech", "cloud", "b2b")):
        return DOMAIN_SAAS
    elif any(k in ind for k in ("hr", "workforce", "human_resources", "talent", "people")):
        return DOMAIN_HR_ATTRITION
    elif any(k in ind for k in ("telecom", "telco", "mobile", "broadband", "cable")):
        return DOMAIN_TELCO

    # 2. Text keyword check in metadata
    text = " ".join(
        str(x or "").lower()
        for x in (project_name, target_column, target_description, organization_industry, metadata_text)
    )

    if any(k in text for k in ("attrition", "employee", "workforce", "turnover", "resign", "hr ")):
        return DOMAIN_HR_ATTRITION
    if any(k in text for k in ("credit", "loan", "lending", "default", "delinquen", "mortgage", "banking", "finance")):
        return DOMAIN_FINANCE
    if any(k in text for k in ("readmission", "patient", "hospital", "clinical", "medical", "doctor", "health")):
        return DOMAIN_HEALTHCARE
    if any(k in text for k in ("ecommerce", "cart", "basket", "shopper", "order_drop", "retail", "checkout")):
        return DOMAIN_ECOMMERCE
    if any(k in text for k in ("saas", "arr", "seat", "software", "license_churn", "api_usage")):
        return DOMAIN_SAAS
    if any(k in text for k in ("churn", "telco", "telecom", "subscriber", "mobile_plan", "cable")):
        return DOMAIN_TELCO

    # 3. Column feature matches
    cols = list(feature_columns or [])
    if not cols and features:
        if isinstance(features, dict):
            cols = list(features.keys())
        elif isinstance(features, (list, tuple, set)):
            cols = list(features)
        else:
            cols = [str(features)]
    norms = {_norm_token(c) for c in cols}

    hr_hits = sum(1 for m in _HR_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    telco_hits = sum(1 for m in _TELCO_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    fin_hits = sum(1 for m in _FINANCE_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    health_hits = sum(1 for m in _HEALTHCARE_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    ecom_hits = sum(1 for m in _ECOMMERCE_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    saas_hits = sum(1 for m in _SAAS_FEATURE_MARKERS if m in norms or any(m in n for n in norms))

    scores = {
        DOMAIN_HR_ATTRITION: hr_hits,
        DOMAIN_TELCO: telco_hits,
        DOMAIN_FINANCE: fin_hits,
        DOMAIN_HEALTHCARE: health_hits,
        DOMAIN_ECOMMERCE: ecom_hits,
        DOMAIN_SAAS: saas_hits,
    }

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] >= 2:
        return best_domain

    if telco_hits >= 1:
        return DOMAIN_TELCO

    return DOMAIN_GENERIC
