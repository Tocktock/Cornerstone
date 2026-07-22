#!/usr/bin/env python3
"""Build and validate the provenance-first VS5 SEC EDGAR evaluation corpus.

The source specification in this file is frozen and intentionally uses only
official ``sec.gov`` filing URLs.  For every source the builder preserves:

* the response bytes returned by SEC EDGAR;
* a deterministic, full-text normalization of the filing HTML; and
* one bounded, verbatim slice used as the CornerStone upload input.

The upload slice is always an exact substring of the normalized full text.
The manifest records its character offsets and hashes all three artifacts.

SEC asks automated clients to identify themselves.  Set ``SEC_USER_AGENT`` to
an identifying product/contact string or pass ``--user-agent``.  The builder
is sequential and rate-limited to at most five requests per second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPOSITORY_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from cornerstone_cli.vs5_corpus import normalize_edgar_filing_html  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "fixtures" / "vs5" / "edgar-eval"
MANIFEST_SCHEMA = "cs.vs5_edgar_eval_manifest.v1"
CORPUS_ID = "vs5-sec-edgar-commercial-contracts-2026-07-17"
MAX_REQUESTS_PER_SECOND = 5.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0 / MAX_REQUESTS_PER_SECOND
DEFAULT_EXTRACT_BYTES = 96 * 1024
MAX_SOURCE_BYTES = 128 * 1024
MAX_CASE_BYTES = 512 * 1024
ALLOWED_SEC_HOSTS = {"sec.gov", "www.sec.gov"}


def _filing_index_url(cik: str, accession: str) -> str:
    accession_compact = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_compact}/{accession}-index.html"
    )


def _source(
    name: str,
    url: str,
    accession: str,
    form: str,
    legal_party: str,
    anchors: Sequence[str],
    *,
    role: str = "contract_chain_document",
    exhibit_number: str = "",
    supersedes: Sequence[str] = (),
    incorporated_by_reference: Sequence[str] = (),
) -> dict[str, Any]:
    path_parts = urllib.parse.urlparse(url).path.split("/")
    try:
        cik = str(int(path_parts[path_parts.index("data") + 1]))
    except (ValueError, IndexError) as error:
        raise ValueError(f"cannot derive CIK from source URL: {url}") from error
    return {
        "name": name,
        "url": url,
        "accession": accession,
        "form": form,
        "legal_party": legal_party,
        "anchors": tuple(anchors),
        "role": role,
        "exhibit_number": exhibit_number,
        "filing_index_url": _filing_index_url(cik, accession),
        "supersedes": tuple(supersedes),
        "incorporated_by_reference": tuple(incorporated_by_reference),
    }


def _case(
    case_id: str,
    issuer: str,
    cik: str,
    decision_question: str,
    sources: Sequence[dict[str, Any]],
    *,
    fact_terms: Sequence[str],
    gap_terms: Sequence[str],
    answerable_question: str,
    answer_terms: Sequence[str],
    unanswerable_question: str,
    contradictions: Sequence[dict[str, Any]] = (),
    relationships: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": case_id,
        "issuer": issuer,
        "cik": cik,
        "archetype": "vendor_contract_renewal",
        "decision_owner": "operational contract owner",
        "operational_decision": decision_question,
        "decision_question": decision_question,
        "sources": tuple(sources),
        "planted_fact_terms": tuple(fact_terms),
        "gap_terms": tuple(gap_terms),
        "contradiction_terms": tuple(
            str(contradiction["term"]) for contradiction in contradictions
        ),
        "contradictions": tuple(contradictions),
        "answerable_question": answerable_question,
        "answer_terms": tuple(answer_terms),
        "unanswerable_question": unanswerable_question,
        "document_relationships": tuple(relationships),
    }


def _contradiction(
    term: str,
    classification: str,
    *,
    prior_source: str,
    prior_claim: str,
    current_source: str,
    current_claim: str,
) -> dict[str, Any]:
    """Declare a real two-sided document change with verbatim source claims."""

    if classification not in {"contradiction", "scope_difference", "supersession"}:
        raise ValueError(f"unsupported contradiction classification: {classification}")
    return {
        "term": term,
        "classification": classification,
        "sides": (
            {
                "side": "prior",
                "source_name": prior_source,
                "claim": prior_claim,
            },
            {
                "side": "current",
                "source_name": current_source,
                "claim": current_claim,
            },
        ),
    }


# The frozen 25-case source plan is populated below.  ``anchors`` are literal,
# case-insensitive phrases used only to position a verbatim upload slice; they
# are not paraphrases or generated source text.
CASES: tuple[dict[str, Any], ...] = (
    _case(
        "edgar-omada-evernorth-msa", "Omada Health, Inc.", "1611115",
        "Should the contract owner continue the Evernorth relationship under the surviving statement-of-work terms or prepare a replacement channel plan?",
        (
            _source("01-evernorth-msa", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex104a.htm", "0001193125-25-116907", "EX-10.4A", "Express Scripts Holding Company, Inc.", ("Term and Termination",), role="master_agreement", exhibit_number="10.4A"),
            _source("02-evernorth-amendment", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex104b.htm", "0001193125-25-116907", "EX-10.4B", "Evernorth Health, Inc.", ("Evernorth",), role="amendment", exhibit_number="10.4B"),
            _source("03-evernorth-amendment", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex104c.htm", "0001193125-25-116907", "EX-10.4C", "Evernorth Health, Inc", ("Evernorth",), role="amendment", exhibit_number="10.4C"),
            _source("04-omada-s1", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770ds1.htm", "0001193125-25-116907", "S-1", "Omada Health, Inc.", ("Evernorth",), role="issuer_disclosure"),
        ),
        fact_terms=("Evernorth", "March 31, 2020"), gap_terms=("pricing", "renewal"),
        answerable_question="What complete deadline wording governs the Corrective Action Plan?", answer_terms=("March 31, 2020 or such later date as is approved by Company in writing",),
        unanswerable_question="What final unredacted price did both parties sign for the next renewal term?",
        relationships=("master agreement -> amendments -> issuer S-1 disclosure",),
    ),
    _case(
        "edgar-composecure-amex-msa", "CompoSecure, Inc.", "1823144",
        "Should the contract owner extend the American Express agreement given its termination, minimum-purchase, and exclusivity provisions?",
        (
            _source("01-amex-agreement", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-22.htm", "0001104659-21-154308", "EX-10.22", "American Express Travel Related Services Company, Inc", ("American Express",), role="master_agreement", exhibit_number="10.22"),
            _source("02-amex-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-26.htm", "0001104659-21-154308", "EX-10.26", "AMERICAN EXPRESS TRAVEL RELATED SERVICES COMPANY, INC.", ("American Express",), role="amendment", exhibit_number="10.26"),
            _source("03-amex-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-29.htm", "0001104659-21-154308", "EX-10.29", "AMERICAN EXPRESS TRAVEL RELATED SERVICES COMPANY, INC.", ("American Express",), role="amendment", exhibit_number="10.29"),
        ),
        fact_terms=("American Express", "[***]", "December 31, 2022"), gap_terms=("volume", "pricing", "exclusivity"),
        answerable_question="On what date did Amendment 4 state that the Initial Term would expire?", answer_terms=("December 31, 2022",),
        unanswerable_question="What are all unredacted minimum-purchase volumes and prices?",
        relationships=("master agreement -> amendments",),
    ),
    _case(
        "edgar-corelogic-ntt-msa", "CoreLogic, Inc.", "36047",
        "Should the contract owner extend the NTT DATA services relationship based on Amendments 4 through 6?",
        (
            _source("01-ntt-amendment-4", "https://www.sec.gov/Archives/edgar/data/36047/000003604718000015/clgx-12312017xex104410k.htm", "0000036047-18-000015", "EX-10.44", "NTT DATA SERVICES, LLC", ("NTT DATA",), role="amendment", exhibit_number="10.44"),
            _source("02-ntt-amendment-5", "https://www.sec.gov/Archives/edgar/data/36047/000003604718000094/ntt_amendmentxno5-ex106.htm", "0000036047-18-000094", "EX-10.6", "NTT DATA SERVICES, LLC", ("NTT",), role="amendment", exhibit_number="10.6"),
            _source("03-ntt-amendment-6", "https://www.sec.gov/Archives/edgar/data/36047/000003604719000086/clgx-63019xex10110q.htm", "0000036047-19-000086", "EX-10.1", "NTT DATA SERVICES, LLC", ("NTT",), role="amendment", exhibit_number="10.1"),
        ),
        fact_terms=("NTT", "October 17, 2018"), gap_terms=("service levels", "pricing", "performance"),
        answerable_question="What effective date is stated for Amendment 6?", answer_terms=("October 17, 2018",),
        unanswerable_question="What were the provider's actual service-level results in the latest measured quarter?",
        relationships=("Amendment 4 -> Amendment 5 -> Amendment 6",),
    ),
    _case(
        "edgar-gogo-airspan-dependency", "Gogo Inc.", "1537054",
        "Should the contract owner continue the Airspan dependency after the Chapter 11 event and June 2024 amendment?",
        (
            _source("01-airspan-agreement", "https://www.sec.gov/Archives/edgar/data/1537054/000156459021012575/gogo-ex1015_112.htm", "0001564590-21-012575", "EX-10.15", "Airspan Networks Inc.", ("Airspan",), role="master_agreement", exhibit_number="10.15"),
            _source("02-airspan-amendment", "https://www.sec.gov/Archives/edgar/data/1537054/000156459021012575/gogo-ex1016_111.htm", "0001564590-21-012575", "EX-10.16", "Airspan Networks Inc.", ("Airspan",), role="amendment", exhibit_number="10.16"),
            _source("03-airspan-2024-amendment", "https://www.sec.gov/Archives/edgar/data/1537054/000095017024092730/gogo-ex10_4.htm", "0000950170-24-092730", "EX-10.4", "Airspan Networks Inc.", ("Airspan",), role="amendment", exhibit_number="10.4"),
            _source("04-gogo-2024-10q", "https://www.sec.gov/Archives/edgar/data/1537054/000095017024092730/gogo-20240630.htm", "0000950170-24-092730", "10-Q", "Gogo Inc.", ("Airspan",), role="issuer_disclosure"),
        ),
        fact_terms=("Airspan", "Chapter 11", "180 Days"), gap_terms=("facility", "conditions", "contingency"),
        answerable_question="How much advance written notice does the renewal amendment require to terminate before a renewal date?", answer_terms=("180 Days",),
        unanswerable_question="What is the tested completion time for a fully independent replacement supply path?",
        relationships=("master agreement -> amendments; issuer disclosure supplies bankruptcy context",),
    ),
    _case(
        "edgar-emergent-astrazeneca-manufacturing", "Emergent BioSolutions Inc.", "1367644",
        "Should the contract owner continue AstraZeneca manufacturing while readiness and remediation evidence remains incomplete?",
        (
            _source("01-az-master", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/a1012-azmaster.htm", "0001367644-20-000163", "EX-10.12", "AstraZeneca Pharmaceuticals LP", ("AstraZeneca",), role="master_agreement", exhibit_number="10.12"),
            _source("02-az-product-schedule", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/a1013-azproductschedule.htm", "0001367644-20-000163", "EX-10.13", "AstraZeneca Pharmaceuticals LP", ("AstraZeneca",), role="product_schedule", exhibit_number="10.13"),
            _source("03-az-amendment", "https://www.sec.gov/Archives/edgar/data/1367644/000136764421000073/emergentbiosolutionsazms.htm", "0001367644-21-000073", "EX-10", "AstraZeneca Pharmaceuticals LP", ("AstraZeneca",), role="amendment", exhibit_number="10"),
            _source("04-ebs-2021-10q", "https://www.sec.gov/Archives/edgar/data/1367644/000136764421000073/ebs-20210331.htm", "0001367644-21-000073", "10-Q", "Emergent BioSolutions Inc.", ("AstraZeneca",), role="issuer_disclosure"),
        ),
        fact_terms=("AstraZeneca", "$174,306,844"), gap_terms=("pricing", "remediation", "regulatory"),
        answerable_question="What total is printed in the Product Schedule's Manufacturing Summary?", answer_terms=("$174,306,844",),
        unanswerable_question="What is the final regulator-approved remediation completion date for every affected batch?",
        relationships=("master agreement -> product schedule -> amendment; 10-Q supplies issuer context",),
    ),
    _case(
        "edgar-itci-lonza-manufacturing", "Intra-Cellular Therapies, Inc.", "1567514",
        "Should the contract owner continue or extend Lonza manufacturing services under the amended agreement?",
        (
            _source("01-lonza-agreement", "https://www.sec.gov/Archives/edgar/data/1567514/000119312520288507/d18633dex101.htm", "0001193125-20-288507", "EX-10.1", "Lonza Ltd", ("Lonza",), role="master_agreement", exhibit_number="10.1"),
            _source("02-lonza-amendment", "https://www.sec.gov/Archives/edgar/data/1567514/000119312523055176/d436875dex1032.htm", "0001193125-23-055176", "EX-10.32", "Lonza Ltd", ("Lonza",), role="amendment", exhibit_number="10.32"),
            _source("03-itci-2022-10k", "https://www.sec.gov/Archives/edgar/data/1567514/000119312523055176/d436875d10k.htm", "0001193125-23-055176", "10-K", "Intra-Cellular Therapies, Inc.", ("Lonza",), role="issuer_disclosure"),
        ),
        fact_terms=("Lonza", "19 day of December 2022"), gap_terms=("capacity", "audit", "performance", "pricing"),
        answerable_question="What date wording does Amendment 1 use for its entry date?", answer_terms=("19 day of December 2022",),
        unanswerable_question="What was the manufacturer's audited on-time batch performance for the latest quarter?",
        relationships=("manufacturing agreement -> amendment -> 10-K issuer disclosure",),
    ),
    _case(
        "edgar-scilex-oishi-itochu-development", "Scilex Holding Company", "1820190",
        "Should the contract owner renegotiate the Oishi and Itochu development relationship or establish alternate supply and intellectual-property arrangements before renewal?",
        (
            _source("01-oishi-development", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659d7_ex10-34.htm", "0001104659-22-081279", "EX-10.34", "Oishi Koseido Co., Ltd.", ("Oishi",), role="development_agreement", exhibit_number="10.34"),
            _source("02-itochu-agreement", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659d7_ex10-38.htm", "0001104659-22-081279", "EX-10.38", "ITOCHU CHEMICAL FRONTIER Corporation", ("Itochu",), role="related_agreement", exhibit_number="10.38"),
            _source(
                "03-itochu-amendment",
                "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659d7_ex10-39.htm",
                "0001104659-22-081279",
                "EX-10.39",
                "ITOCHU CHEMICAL FRONTIER Corporation",
                ("Itochu",),
                role="amendment",
                exhibit_number="10.39",
                supersedes=("02-itochu-agreement",),
                incorporated_by_reference=("01-oishi-development",),
            ),
            _source("04-scilex-s4a", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659-6_s4a.htm", "0001104659-22-081279", "S-4/A", "Scilex Holding Company", ("Oishi", "Itochu"), role="issuer_disclosure"),
        ),
        fact_terms=("Oishi", "Itochu", "cancel the effect of the Fourth Amendment to the Development Agreement", "five percent (5%)"), gap_terms=("alternate supply",),
        answerable_question="What prior amendment effect do the parties state they are cancelling in the Fifth Amendment?", answer_terms=("cancel the effect of the Fourth Amendment to the Development Agreement",),
        unanswerable_question="What signed commercial terms would an alternate supplier offer?",
        contradictions=(
            _contradiction(
                "Fifth Amendment cancels the Fourth Amendment",
                "supersession",
                prior_source="02-itochu-agreement",
                prior_claim="If at any time commencing with the quarter ending March 31, 2023",
                current_source="03-itochu-amendment",
                current_claim="If at any time during the Term",
            ),
        ),
        relationships=(
            "development agreement -> Fourth Amendment -> Fifth Amendment; Fifth Amendment cancels the Fourth Amendment; S-4/A supplies issuer context",
        ),
    ),
    _case(
        "edgar-savara-gema-supply", "Savara Inc.", "1160308",
        "Should the contract owner remain dependent on GEMA manufacturing or accelerate a qualified second source?",
        (
            _source("01-gema-agreement", "https://www.sec.gov/Archives/edgar/data/1160308/000156459019017983/svra-ex101_531.htm", "0001564590-19-017983", "EX-10.1", "GEMABIOTECH SAU", ("GEMA",), role="master_agreement", exhibit_number="10.1"),
            _source("02-gema-amendment", "https://www.sec.gov/Archives/edgar/data/1160308/000095017023010967/svra-ex10_33.htm", "0000950170-23-010967", "EX-10.33", "GEMABIOTECH SAU", ("GEMA",), role="amendment", exhibit_number="10.33"),
            _source("03-savara-2022-10k", "https://www.sec.gov/Archives/edgar/data/1160308/000095017023010967/svra-20221231.htm", "0000950170-23-010967", "10-K", "Savara Inc.", ("GEMA",), role="issuer_disclosure"),
        ),
        fact_terms=("GEMA", "twentieth (20th) anniversary"), gap_terms=("validation", "inspection", "second source"),
        answerable_question="According to Section 13.1, what event marks the end of the original GEMA Agreement's Initial Term?",
        answer_terms=("twentieth (20th) anniversary of the date of receipt of approval by a Regulatory Authority of the first Regulatory Filing for the marketing and sale of the first Product in any country",),
        unanswerable_question="Which second source has completed validation and demonstrated regulatory equivalence?",
        relationships=("manufacturing agreement -> amendment -> 10-K issuer disclosure",),
    ),
    _case(
        "edgar-healthnet-ibm-outsourcing", "Health Net, Inc.", "916085",
        "As of the latest packet filing, should the contract owner continue IBM outsourcing or prepare a transition after the missing-drive incident?",
        (
            _source("01-ibm-outsourcing-agreement", "https://www.sec.gov/Archives/edgar/data/916085/000119312508230367/dex101.htm", "0001193125-08-230367", "EX-10.1", "International Business Machines Corporation", ("IBM",), role="outsourcing_agreement", exhibit_number="10.1"),
            _source("02-healthnet-2008-10k", "https://www.sec.gov/Archives/edgar/data/916085/000119312509039486/d10k.htm", "0001193125-09-039486", "10-K", "Health Net, Inc.", ("IBM",), role="issuer_disclosure"),
            _source("03-healthnet-2012-10k", "https://www.sec.gov/Archives/edgar/data/916085/000091608513000004/hnt201210k.htm", "0000916085-13-000004", "10-K", "Health Net, Inc.", ("IBM",), role="issuer_disclosure"),
        ),
        fact_terms=("IBM", "2 million", "February 14, 2014"), gap_terms=("root cause", "remediation", "exit cost"),
        answerable_question="On what date does the Master Agreement state that its Initial Term expires?", answer_terms=("February 14, 2014",),
        unanswerable_question="What independently verified root cause and completed remediation record closed the missing-drive incident?",
        relationships=("outsourcing agreement; later 10-K disclosures supply operational and incident context",),
    ),
    _case(
        "edgar-iteos-gsk-collaboration", "iTeos Therapeutics, Inc.", "1808865",
        "Should the contract owner continue iTeos's share of the GSK global development plan under the amended collaboration?",
        (
            _source("01-gsk-collaboration", "https://www.sec.gov/Archives/edgar/data/1808865/000156459021043439/itos-ex101_116.htm", "0001564590-21-043439", "EX-10.1", "GLAXOSMITHKLINE INTELLECTUAL PROPERTY (No. 4) LIMITED", ("GSK",), role="collaboration_agreement", exhibit_number="10.1"),
            _source("02-gsk-amendment", "https://www.sec.gov/Archives/edgar/data/1808865/000095017023007971/itos-ex10_14.htm", "0000950170-23-007971", "EX-10.14", "GLAXOSMITHKLINE INTELLECTUAL PROPERTY (No. 4) LIMITED", ("GSK",), role="amendment", exhibit_number="10.14"),
            _source("03-gsk-amendment", "https://www.sec.gov/Archives/edgar/data/1808865/000095017023007971/itos-ex10_15.htm", "0000950170-23-007971", "EX-10.15", "GLAXOSMITHKLINE INTELLECTUAL PROPERTY (No. 4) LIMITED", ("GSK",), role="amendment", exhibit_number="10.15"),
            _source("04-iteos-2022-10k", "https://www.sec.gov/Archives/edgar/data/1808865/000095017023007971/itos-20221231.htm", "0000950170-23-007971", "10-K", "iTeos Therapeutics, Inc.", ("$900 million", "GSK is responsible for 60%"), role="issuer_disclosure"),
        ),
        fact_terms=("$625.0 million", "$900 million", "sixty percent (60%)", "forty percent (40%)"), gap_terms=("trial outcomes", "milestones"),
        contradictions=(
            _contradiction(
                "Amendment No. 1 excludes products containing non-Licensed-Antibody "
                "ITEOS/Affiliate-owned or controlled ingredients from GSK's Licensed "
                "Product scope",
                "scope_difference",
                prior_source="01-gsk-collaboration",
                prior_claim="“Licensed Product” means any pharmaceutical product that is comprised of or contains a Licensed Antibody",
                current_source="02-gsk-amendment",
                current_claim="Licensed Product does not include, and GSK is not granted right to, any pharmaceutical product that is comprised of or contains any compound, antibody, or other pharmaceutically active ingredient owned or Controlled by ITEOS or any of its Affiliates, in each case, that is not a Licensed Antibody.",
            ),
        ),
        answerable_question="What development-cost split, activity scope, and stated exceptions govern the collaboration agreement?",
        answer_terms=(
            "Shared Global Development Activities",
            "sixty percent (60%)",
            "forty percent (40%)",
            "Section 3.4 (Additional Development)",
            "Section 6.7 (ITEOS Opt-Out)",
        ),
        unanswerable_question="Which future clinical milestones will ultimately succeed and on what dates?",
        relationships=("collaboration agreement -> amendments; 10-K supplies issuer disclosure",),
    ),
    _case(
        "edgar-omada-cigna-services", "Omada Health, Inc.", "1611115",
        "Should the contract owner continue or renew the Cigna services arrangement under the controlling agreement and amendment?",
        (
            _source("01-cigna-services-agreement", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex103a.htm", "0001193125-25-116907", "EX-10.3A", "Cigna Health and Life Insurance Company", ("Cigna",), role="master_agreement", exhibit_number="10.3A"),
            _source("02-cigna-services-amendment", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex103b.htm", "0001193125-25-116907", "EX-10.3B", "Cigna Health and Life Insurance Company", ("Cigna",), role="amendment", exhibit_number="10.3B"),
            _source("03-omada-s1", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770ds1.htm", "0001193125-25-116907", "S-1", "Omada Health, Inc.", ("Cigna",), role="issuer_disclosure"),
        ),
        fact_terms=("Cigna", "May 30, 2018"), gap_terms=("pricing", "renewal", "service performance"),
        answerable_question="What date is shown for Cigna's signatory in the original services agreement?", answer_terms=("May 30, 2018",),
        unanswerable_question="What final signed price and measured service performance will govern the next renewal?",
        relationships=("services agreement -> amendment; S-1 supplies issuer disclosure",),
    ),
    _case(
        "edgar-omada-cigna-admin", "Omada Health, Inc.", "1611115",
        "Should the contract owner continue or renew the Cigna administrative-services arrangement under the latest controlling amendment?",
        (
            _source("01-cigna-admin-agreement", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex105a.htm", "0001193125-25-116907", "EX-10.5A", "Cigna Health and Life Insurance Company", ("Cigna",), role="administrative_services_agreement", exhibit_number="10.5A"),
            _source("02-cigna-admin-amendment", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex105b.htm", "0001193125-25-116907", "EX-10.5B", "Cigna Health and Life Insurance Company", ("Cigna",), role="amendment", exhibit_number="10.5B"),
            _source("03-cigna-admin-amendment", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770dex105c.htm", "0001193125-25-116907", "EX-10.5C", "Cigna Health and Life Insurance Company", ("Cigna",), role="amendment", exhibit_number="10.5C"),
            _source("04-omada-s1", "https://www.sec.gov/Archives/edgar/data/1611115/000119312525116907/d785770ds1.htm", "0001193125-25-116907", "S-1", "Omada Health, Inc.", ("Cigna",), role="issuer_disclosure"),
        ),
        fact_terms=("Cigna", "March 7,\n2022"), gap_terms=("pricing", "service levels", "renewal"),
        answerable_question="What effective date is stated for Administrative Services Agreement Amendment No. 2?", answer_terms=("March 7,\n2022",),
        unanswerable_question="What complete unredacted economics and measured service levels will govern the next term?",
        relationships=("administrative-services agreement -> amendments; S-1 supplies issuer disclosure",),
    ),
    _case(
        "edgar-sema4-mount-sinai-services", "Sema4 Holdings Corp.", "1818331",
        "Should the contract owner continue the Mount Sinai services arrangement after applying the amendment effects?",
        (
            _source("01-mount-sinai-services", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1022-super8xk.htm", "0001628280-21-014760", "EX-10.22", "Icahn School of Medicine at Mount Sinai", ("Mount Sinai",), role="services_agreement", exhibit_number="10.22"),
            _source("02-mount-sinai-services-amendment", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1023-super8xk.htm", "0001628280-21-014760", "EX-10.23", "Icahn School of Medicine at Mount Sinai", ("Mount Sinai",), role="amendment", exhibit_number="10.23"),
            _source("03-sema4-super8k", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/super8-k.htm", "0001628280-21-014760", "8-K", "Sema4 Holdings Corp.", ("Mount Sinai",), role="issuer_disclosure"),
        ),
        fact_terms=("Mount Sinai", "three (3) years"), gap_terms=("performance", "pricing", "renewal"),
        answerable_question="How long is the Initial Term in the second Mount Sinai services agreement?", answer_terms=("three (3) years",),
        unanswerable_question="What independently measured service performance and signed next-term price support continuation?",
        relationships=("services agreement -> amendment; 8-K supplies issuer disclosure",),
    ),
    _case(
        "edgar-sema4-mount-sinai-data-rights", "Sema4 Holdings Corp.", "1818331",
        "Should the contract owner continue the Mount Sinai data-rights arrangement given the operative intellectual-property, license, and post-termination constraints?",
        (
            _source("01-mount-sinai-data-agreement", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1024-super8xk.htm", "0001628280-21-014760", "EX-10.24", "Icahn School of Medicine at Mount Sinai", ("Mount Sinai",), role="data_rights_agreement", exhibit_number="10.24"),
            _source("02-mount-sinai-data-amendment", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1025-super8xk.htm", "0001628280-21-014760", "EX-10.25", "Icahn School of Medicine at Mount Sinai", ("Mount Sinai",), role="amendment", exhibit_number="10.25"),
            _source("03-mount-sinai-related-agreement", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1026-super8xk.htm", "0001628280-21-014760", "EX-10.26", "Icahn School of Medicine at Mount Sinai", ("Mount Sinai",), role="related_agreement", exhibit_number="10.26"),
            _source("04-sema4-super8k", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/super8-k.htm", "0001628280-21-014760", "8-K", "Sema4 Holdings Corp.", ("Mount Sinai",), role="issuer_disclosure"),
        ),
        fact_terms=("Mount Sinai", "October 12, 2018"), gap_terms=("post-termination", "data rights", "pricing"),
        answerable_question="What date is stated for the referenced JCAP Statement of Work No. 2?", answer_terms=("October 12, 2018",),
        unanswerable_question="What signed future arrangement resolves every post-termination data-use dispute?",
        relationships=("data-rights agreement -> amendment + related agreement; 8-K supplies issuer disclosure",),
    ),
    _case(
        "edgar-sema4-illumina-supply", "Sema4 Holdings Corp.", "1818331",
        "Should the contract owner continue the Illumina supply dependency given the disclosed agreement and remaining continuity gaps?",
        (
            _source("01-illumina-supply", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit1027-super8xk.htm", "0001628280-21-014760", "EX-10.27", "Illumina, Inc.", ("Illumina",), role="supply_agreement", exhibit_number="10.27"),
            _source("02-sema4-super8k", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/super8-k.htm", "0001628280-21-014760", "8-K", "Sema4 Holdings Corp.", ("Illumina",), role="issuer_disclosure"),
            _source("03-sema4-exhibit-991", "https://www.sec.gov/Archives/edgar/data/1818331/000162828021014760/exhibit991-super8xk.htm", "0001628280-21-014760", "EX-99.1", "Sema4 Holdings Corp", ("Sema4",), role="issuer_communication", exhibit_number="99.1"),
        ),
        fact_terms=("Illumina", "August 20, 2014"), gap_terms=("pricing", "volume", "continuity"),
        contradictions=(
            _contradiction(
                "First Amendment date conflicts with exhibit index Supply Agreement date",
                "contradiction",
                prior_source="02-sema4-super8k",
                prior_claim="Supply Agreement, dated as of June 20, 2014, by and between the Company and Illumina, Inc., and amendments thereto.",
                current_source="01-illumina-supply",
                current_claim="WHEREAS, the Parties entered into a Supply Agreement, dated August 20, 2014 (“Agreement”);",
            ),
        ),
        answerable_question="According to the First Amendment recital, what date does it state for the original Illumina Supply Agreement?", answer_terms=("August 20, 2014",),
        unanswerable_question="What fully qualified alternate supply capacity is available on the same commercial terms?",
        relationships=("supply agreement; 8-K and exhibit 99.1 supply issuer context",),
    ),
    _case(
        "edgar-composecure-jpm-msa", "CompoSecure, Inc.", "1823144",
        "Should the contract owner continue the JPMorgan agreement chain while managing concentration and undisclosed-economics risk?",
        (
            _source("01-jpm-agreement", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-30.htm", "0001104659-21-154308", "EX-10.30", "JPMorgan Chase Bank, National Association", ("JPMorgan",), role="master_agreement", exhibit_number="10.30"),
            _source("02-jpm-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-31.htm", "0001104659-21-154308", "EX-10.31", "JPMORGAN CHASE BANK, NATIONAL ASSOCIATION", ("JPMorgan",), role="amendment", exhibit_number="10.31"),
            _source("03-jpm-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-32.htm", "0001104659-21-154308", "EX-10.32", "JPMorgan Chase Bank, National Association", ("JPMorgan",), role="amendment", exhibit_number="10.32"),
            _source("04-jpm-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-33.htm", "0001104659-21-154308", "EX-10.33", "JPMorgan Chase Bank, N.A.", ("JPMorgan",), role="amendment", exhibit_number="10.33"),
        ),
        fact_terms=("JPMorgan", "[***]", "May 1, 2014"), gap_terms=("economics", "concentration", "performance"),
        answerable_question="What Effective Date is printed in Amendment CW673842 to Master Services Agreement CW232350?", answer_terms=("May 1, 2014",),
        unanswerable_question="What are the complete unredacted future volumes, prices, and realized margins?",
        relationships=("master agreement -> amendments",),
    ),
    _case(
        "edgar-composecure-facility-lease", "CompoSecure, Inc.", "1823144",
        "Should the facilities owner renew or exit the leased site under the amendment-controlled term and occupancy exposure?",
        (
            _source("01-facility-lease", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-20.htm", "0001104659-21-154308", "EX-10.20", "BAKER-PROPERTIES LIMITED PARTNERSHIP", ("Lease",), role="lease", exhibit_number="10.20"),
            _source("02-facility-lease-amendment", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_ex10-21.htm", "0001104659-21-154308", "EX-10.21", "BAKER-PROPERTIES LIMITED PARTNERSHIP", ("Lease",), role="lease_amendment", exhibit_number="10.21"),
            _source("03-composecure-8k", "https://www.sec.gov/Archives/edgar/data/1823144/000110465921154308/tm2135648d1_8k.htm", "0001104659-21-154308", "8-K", "CompoSecure, Inc.", ("lease",), role="issuer_disclosure"),
        ),
        fact_terms=("Lease", "March 31, 2027"), gap_terms=("occupancy", "exit cost", "renewal"),
        answerable_question="To what date does the lease amendment extend the Term?", answer_terms=("March 31, 2027",),
        unanswerable_question="What independently appraised market rent and approved future occupancy plan support renewal?",
        relationships=("lease -> lease amendment; 8-K supplies issuer context",),
    ),
    _case(
        "edgar-itci-bms-license", "Intra-Cellular Therapies, Inc.", "1567514",
        "Should the contract owner continue the Bristol-Myers Squibb license under the amended scope and termination terms?",
        (
            _source("01-bms-license", "https://www.sec.gov/Archives/edgar/data/1567514/000119312513358382/d590040dex1011.htm", "0001193125-13-358382", "EX-10.11", "Bristol-Myers Squibb Company", ("Bristol-Myers",), role="license_agreement", exhibit_number="10.11"),
            _source("02-bms-license-amendment", "https://www.sec.gov/Archives/edgar/data/1567514/000119312513358382/d590040dex1012.htm", "0001193125-13-358382", "EX-10.12", "Bristol-Myers Squibb Company", ("Bristol-Myers",), role="amendment", exhibit_number="10.12", supersedes=("01-bms-license",)),
            _source("03-itci-8k", "https://www.sec.gov/Archives/edgar/data/1567514/000119312513358382/d590040d8k.htm", "0001193125-13-358382", "8-K", "Intra-Cellular Therapies, Inc.", ("Bristol-Myers",), role="issuer_disclosure"),
        ),
        fact_terms=("Bristol-Myers", "entered into effective November 3, 2010"), gap_terms=("territory", "scope", "termination"),
        contradictions=(
            _contradiction(
                "Amendment No. 1 deletes ITI Qualified Study prerequisite before "
                "third-party licensing",
                "supersession",
                prior_source="01-bms-license",
                prior_claim="ITI shall not enter into a License Agreement, or enter into discussions with any Third Party with respect to any License,\nuntil Completion of the Qualified Study except as otherwise provided for in Section 2.3.1",
                current_source="02-bms-license-amendment",
                current_claim="The Parties hereby amend the Agreement (i) to delete the requirement that ITI complete a Qualified Study before pursuing any License\nwith a Third Party",
            ),
        ),
        answerable_question="What effective-date wording appears in Amendment No. 1?", answer_terms=("effective November 3, 2010",),
        unanswerable_question="What signed future amendment resolves every territory, scope, and termination uncertainty?",
        relationships=("license agreement -> amendment; 8-K supplies issuer context",),
    ),
    _case(
        "edgar-itci-siegfried-supply", "Intra-Cellular Therapies, Inc.", "1567514",
        "Should the contract owner continue the Siegfried manufacturing and supply dependency?",
        (
            _source("01-siegfried-supply", "https://www.sec.gov/Archives/edgar/data/1567514/000119312523055176/d436875dex102.htm", "0001193125-23-055176", "EX-10.2", "Siegfried AG", ("Siegfried",), role="supply_agreement", exhibit_number="10.2"),
            _source("02-itci-2022-10k", "https://www.sec.gov/Archives/edgar/data/1567514/000119312523055176/d436875d10k.htm", "0001193125-23-055176", "10-K", "Intra-Cellular Therapies, Inc.", ("Siegfried",), role="issuer_disclosure"),
        ),
        fact_terms=("Siegfried", "January 5, 2026"), gap_terms=("price", "capacity", "quality"),
        answerable_question="Until what date does the 10-K say the initial term of the Siegfried Agreement runs?", answer_terms=("January 5, 2026",),
        unanswerable_question="What audited future capacity and quality performance is guaranteed at an unredacted price?",
        relationships=("supply agreement; 10-K supplies issuer context",),
    ),
    _case(
        "edgar-emergent-az-workorder", "Emergent BioSolutions Inc.", "1367644",
        "Should the contract owner authorize continued AstraZeneca work-order performance as aligned with the governing master agreement?",
        (
            _source("01-az-work-order", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/a1014-azminixmsa.htm", "0001367644-20-000163", "EX-10.14", "AstraZeneca Pharmaceuticals LP", ("AstraZeneca",), role="work_order", exhibit_number="10.14"),
            _source("02-az-master", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/a1012-azmaster.htm", "0001367644-20-000163", "EX-10.12", "AstraZeneca Pharmaceuticals LP", ("AstraZeneca",), role="master_agreement", exhibit_number="10.12"),
            _source("03-ebs-2020-10q", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/ebs-20200930.htm", "0001367644-20-000163", "10-Q", "Emergent BioSolutions Inc.", ("AstraZeneca",), role="issuer_disclosure"),
        ),
        fact_terms=("AstraZeneca", "$87,453,649"), gap_terms=("acceptance", "performance", "pricing"),
        answerable_question="What subtotal is printed for Drug Substance Manufacturing and PPQ in the work order?", answer_terms=("$87,453,649",),
        unanswerable_question="What completed acceptance evidence proves every work-order deliverable met its specification?",
        relationships=("work order governed by master agreement; 10-Q supplies issuer context",),
    ),
    _case(
        "edgar-emergent-janssen-manufacturing", "Emergent BioSolutions Inc.", "1367644",
        "Should the contract owner continue Janssen manufacturing under the obligations as changed by the later amendment?",
        (
            _source("01-janssen-agreement", "https://www.sec.gov/Archives/edgar/data/1367644/000136764420000163/a1016-janssenagreement.htm", "0001367644-20-000163", "EX-10.16", "Janssen Pharmaceuticals, Inc.", ("Janssen",), role="manufacturing_agreement", exhibit_number="10.16"),
            _source("02-janssen-amendment", "https://www.sec.gov/Archives/edgar/data/1367644/000136764421000073/emergentbiosolutionsjans.htm", "0001367644-21-000073", "EX-10", "Janssen Pharmaceuticals, Inc.", ("Janssen",), role="amendment", exhibit_number="10"),
            _source("03-ebs-2021-10q", "https://www.sec.gov/Archives/edgar/data/1367644/000136764421000073/ebs-20210331.htm", "0001367644-21-000073", "10-Q", "Emergent BioSolutions Inc.", ("Janssen",), role="issuer_disclosure"),
        ),
        fact_terms=("Janssen", "January 11, 2021"), gap_terms=("performance", "remediation", "pricing"),
        answerable_question="On what date does Contract Year 1 begin under the Janssen amendment?", answer_terms=("January 11, 2021",),
        unanswerable_question="What final completed remediation and batch-acceptance record supports every future obligation?",
        relationships=("manufacturing agreement -> later amendment; 10-Q supplies issuer context; temporal changes are versioning",),
    ),
    _case(
        "edgar-gogo-hughes-msa", "Gogo Inc.", "1537054",
        "Should the contract owner continue the Hughes service-provider dependency under the disclosed agreement and later relationship status?",
        (
            _source("01-hughes-msa", "https://www.sec.gov/Archives/edgar/data/1537054/000095017022010727/gogo-ex10_1.htm", "0000950170-22-010727", "EX-10.1", "Hughes Network Systems, LLC", ("Hughes",), role="master_agreement", exhibit_number="10.1"),
            _source("02-gogo-2022-filing", "https://www.sec.gov/Archives/edgar/data/1537054/000095017022010727/gogo-20220526.htm", "0000950170-22-010727", "8-K", "Gogo Inc.", ("Hughes",), role="issuer_disclosure"),
            _source("03-gogo-2024-10q", "https://www.sec.gov/Archives/edgar/data/1537054/000095017024092730/gogo-20240630.htm", "0000950170-24-092730", "10-Q", "Gogo Inc.", ("satellite providers",), role="issuer_disclosure"),
        ),
        fact_terms=("Hughes", "made and entered into as of May 21, 2022"), gap_terms=("service levels", "transition", "performance"),
        answerable_question="What effective-date wording appears in the Hughes Master Services Agreement?", answer_terms=("made and entered into as of May 21, 2022",),
        unanswerable_question="What tested transition plan and current measured service levels support replacing the provider?",
        relationships=("master agreement and 2022 issuer disclosure name Hughes; the 2024 10-Q supplies broader satellite-provider dependency context",),
    ),
    _case(
        "edgar-scilex-lifecore-msa", "Scilex Holding Company", "1820190",
        "Should the contract owner continue Lifecore manufacturing and supply under the amendment-controlled obligations?",
        (
            _source("01-lifecore-msa", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922074469/vcka-20220331xex10d40.htm", "0001104659-22-074469", "EX-10.40", "Lifecore Biomedical, LLC", ("Lifecore",), role="manufacturing_agreement", exhibit_number="10.40"),
            _source("02-lifecore-amendment", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922074469/vcka-20220331xex10d41.htm", "0001104659-22-074469", "EX-10.41", "Lifecore Biomedical, LLC", ("Lifecore",), role="amendment", exhibit_number="10.41", supersedes=("01-lifecore-msa",)),
            _source("03-scilex-s4a", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659-6_s4a.htm", "0001104659-22-081279", "S-4/A", "Scilex Holding Company", ("Lifecore",), role="issuer_disclosure"),
        ),
        fact_terms=("Lifecore", "December 31, 2022"), gap_terms=("capacity", "quality", "pricing"),
        contradictions=(
            _contradiction(
                "Amendment No. 1 replaces earlier-of services-completion endpoint",
                "supersession",
                prior_source="01-lifecore-msa",
                prior_claim="will continue until the earlier of (i) completion of the services in Exhibit B, or (ii) December 31, 2020",
                current_source="02-lifecore-amendment",
                current_claim="Subsection 7.1 is amended by replacing the words, “the earlier of (i) completion of the services in Exhibit B, or (ii) December 31, 2020” with the words, “December 31, 2022.”",
            ),
        ),
        answerable_question="To what date does the Lifecore amendment extend the services term?", answer_terms=("December 31, 2022",),
        unanswerable_question="What audited future capacity and quality record is guaranteed at a complete unredacted price?",
        relationships=("manufacturing agreement -> amendment; S-4/A supplies issuer context",),
    ),
    _case(
        "edgar-scilex-tulex-msa", "Scilex Holding Company", "1820190",
        "Should the contract owner continue the Tulex services and manufacturing relationship under the amended responsibilities?",
        (
            _source("01-tulex-msa", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922074469/vcka-20220331xex10d42.htm", "0001104659-22-074469", "EX-10.42", "Tulex", ("Tulex",), role="master_agreement", exhibit_number="10.42"),
            _source("02-tulex-amendment", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922074469/vcka-20220331xex10d43.htm", "0001104659-22-074469", "EX-10.43", "TULEX", ("Tulex",), role="amendment", exhibit_number="10.43"),
            _source("03-scilex-s4a", "https://www.sec.gov/Archives/edgar/data/1820190/000110465922081279/tm2214659-6_s4a.htm", "0001104659-22-081279", "S-4/A", "Scilex Holding Company", ("Tulex",), role="issuer_disclosure"),
        ),
        fact_terms=("Tulex", "June 15, 2022"), gap_terms=("performance", "capacity", "pricing"),
        answerable_question="What Effective Date is stated for the Tulex Novation Agreement?", answer_terms=("June 15, 2022",),
        unanswerable_question="What audited future capacity and service performance supports continuation at a complete unredacted price?",
        relationships=("master agreement -> amendment; S-4/A supplies issuer context",),
    ),
    _case(
        "edgar-healthnet-cognizant-expansion", "Health Net, Inc.", "916085",
        "Should the contract owner continue the expanded Cognizant outsourcing scope and dependency under the amended arrangement?",
        (
            _source("01-cognizant-agreement", "https://www.sec.gov/Archives/edgar/data/916085/000119312508230367/dex102.htm", "0001193125-08-230367", "EX-10.2", "Cognizant Technology Solutions U.S. Corporation", ("Cognizant",), role="outsourcing_agreement", exhibit_number="10.2"),
            _source("02-cognizant-amendment", "https://www.sec.gov/Archives/edgar/data/916085/000144530512003449/exhibit101amendment3-hnfsr.htm", "0001445305-12-003449", "EX-10.1", "Cognizant Technology Solutions U.S. Corporation", ("Cognizant",), role="amendment", exhibit_number="10.1"),
            _source("03-healthnet-2014-10k", "https://www.sec.gov/Archives/edgar/data/916085/000091608515000004/hnt201410k.htm", "0000916085-15-000004", "10-K", "Health Net, Inc.", ("Cognizant",), role="issuer_disclosure"),
        ),
        fact_terms=("Cognizant", "$10 million"), gap_terms=("exit cost", "performance", "transition"),
        answerable_question="What break-up fee does the 10-K say applies if Health Net terminates before the Commencement Date?", answer_terms=("$10 million",),
        unanswerable_question="What tested transition duration and fully quantified exit cost support replacing the expanded outsourcing scope?",
        relationships=("outsourcing agreement -> amendment; later 10-K supplies dependency context",),
    ),
)

HUMAN_REVIEW_CASE_IDS = (
    "edgar-omada-evernorth-msa",
    "edgar-composecure-amex-msa",
    "edgar-corelogic-ntt-msa",
    "edgar-gogo-airspan-dependency",
    "edgar-emergent-astrazeneca-manufacturing",
    "edgar-itci-lonza-manufacturing",
    "edgar-scilex-oishi-itochu-development",
    "edgar-savara-gema-supply",
    "edgar-healthnet-ibm-outsourcing",
    "edgar-iteos-gsk-collaboration",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write_bytes_if_changed(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == value:
        return
    path.write_bytes(value)


def _write_text_if_changed(path: Path, value: str) -> None:
    _write_bytes_if_changed(path, value.encode("utf-8"))


def _literal_positions(text: str, anchors: Sequence[str]) -> list[int]:
    lowered = text.casefold()
    positions: list[int] = []
    for anchor in anchors:
        position = lowered.find(anchor.casefold())
        if position < 0:
            raise ValueError(f"required extraction anchor not found: {anchor!r}")
        positions.append(position)
    return positions


def _utf8_bounded_end(text: str, start: int, max_bytes: int) -> int:
    """Find the largest character end whose UTF-8 slice fits ``max_bytes``."""

    low = start
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(text[start:middle].encode("utf-8")) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return low


def select_verbatim_extract(
    normalized: str,
    *,
    anchors: Sequence[str],
    max_bytes: int,
) -> tuple[str, int, int]:
    """Select one exact, line-bounded substring containing all source anchors."""

    positions = _literal_positions(normalized, anchors)
    anchor_start = min(positions)
    anchor_end = max(
        position + len(anchor)
        for position, anchor in zip(positions, anchors, strict=True)
    )
    if len(normalized[anchor_start:anchor_end].encode("utf-8")) > max_bytes:
        raise ValueError("required anchors cannot fit in one bounded upload extract")

    remaining_chars = max_bytes - len(normalized[anchor_start:anchor_end].encode("utf-8"))
    approximate_side = remaining_chars // 2
    start = max(0, anchor_start - approximate_side)
    line_start = normalized.find("\n", start, anchor_start)
    if line_start >= 0:
        start = line_start + 1
    end = _utf8_bounded_end(normalized, start, max_bytes)
    if end < anchor_end:
        start = max(0, anchor_end - max_bytes)
        line_start = normalized.find("\n", start, anchor_start)
        if line_start >= 0:
            start = line_start + 1
        end = _utf8_bounded_end(normalized, start, max_bytes)
    line_end = normalized.rfind("\n", anchor_end, end)
    if line_end >= anchor_end:
        end = line_end + 1
    extract = normalized[start:end]
    if not extract or normalized[start:end] != extract:
        raise AssertionError("extract is not an exact normalized-text span")
    for anchor in anchors:
        if anchor.casefold() not in extract.casefold():
            raise AssertionError(f"extract omitted required anchor: {anchor!r}")
    return extract, start, end


class SecClient:
    def __init__(self, *, user_agent: str) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT/--user-agent must identify the client and include a contact email"
            )
        self.user_agent = user_agent.strip()
        self._last_request_started = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_started
        delay = MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if delay > 0:
            time.sleep(delay)
        self._last_request_started = time.monotonic()

    def get(self, url: str, *, attempts: int = 4) -> tuple[bytes, dict[str, str]]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SEC_HOSTS:
            raise ValueError(f"source URL is not an official HTTPS SEC URL: {url}")
        for attempt in range(1, attempts + 1):
            self._rate_limit()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "identity",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = int(response.status)
                    final_url = response.geturl()
                    final_host = urllib.parse.urlparse(final_url).hostname
                    if status != 200:
                        raise RuntimeError(f"SEC returned HTTP {status}: {url}")
                    if final_host not in ALLOWED_SEC_HOSTS:
                        raise RuntimeError(f"SEC URL redirected outside sec.gov: {final_url}")
                    body = response.read()
                    return body, {
                        "http_status": str(status),
                        "final_url": final_url,
                        "content_type": response.headers.get("Content-Type", ""),
                        "last_modified": response.headers.get("Last-Modified", ""),
                        "etag": response.headers.get("ETag", ""),
                    }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                if attempt >= attempts:
                    raise RuntimeError(f"failed to fetch {url}: {error}") from error
                time.sleep(min(2 ** (attempt - 1), 8))
        raise AssertionError("unreachable")


def _safe_filename(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
        raise ValueError(f"unsafe corpus filename: {value!r}")
    return value


def _legacy_source_paths(
    output_root: Path, case_id: str, source: dict[str, Any]
) -> dict[str, Path]:
    basename = _safe_filename(str(source["name"]))
    return {
        "raw": output_root / case_id / "raw" / f"{basename}.html",
        "normalized": output_root / case_id / "normalized" / f"{basename}.txt",
        "upload": output_root / case_id / "upload" / f"{basename}.txt",
    }


def _shared_artifact_path(
    output_root: Path, *, kind: str, sha256: str, suffix: str
) -> Path:
    if kind not in {"raw", "normalized"}:
        raise ValueError(f"unsupported shared artifact kind: {kind}")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"invalid content hash: {sha256!r}")
    return output_root / "_shared" / kind / "sha256" / f"{sha256}{suffix}"


def _previous_artifact_path(
    metadata: dict[str, Any] | None,
    *,
    key: str,
    output_root: Path,
) -> Path | None:
    if not metadata or not metadata.get(key):
        return None
    path = REPOSITORY_ROOT / str(metadata[key])
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(f"previous manifest path escapes corpus root: {metadata[key]}") from error
    return path if path.is_file() else None


def _remove_legacy_source_copies(output_root: Path) -> None:
    """Remove superseded case-local full copies after shared objects are durable."""

    for case in CASES:
        case_root = output_root / str(case["id"])
        for directory_name in ("raw", "normalized"):
            directory = case_root / directory_name
            if directory.is_dir():
                shutil.rmtree(directory)


def _remove_unreferenced_shared_objects(
    output_root: Path, manifest: dict[str, Any]
) -> None:
    referenced = {
        (REPOSITORY_ROOT / str(source[path_key])).resolve()
        for case in manifest["cases"]
        for source in case["sources"]
        for path_key in ("raw_path", "normalized_path")
    }
    shared_root = output_root / "_shared"
    if not shared_root.is_dir():
        return
    for path in shared_root.rglob("*"):
        if path.is_file() and path.resolve() not in referenced:
            path.unlink()


def _relative(path: Path, output_root: Path) -> str:
    del output_root
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _load_existing_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_source_metadata(
    manifest: dict[str, Any], case_id: str, source_name: str
) -> dict[str, Any] | None:
    for case in manifest.get("cases", []):
        if case.get("id") != case_id:
            continue
        for source in case.get("sources", []):
            if source.get("name") == source_name:
                return source
    return None


def _read_response_cache(cache_root: Path, url: str) -> tuple[bytes, dict[str, str]]:
    key = _sha256_text(url)
    raw_path = cache_root / f"{key}.html"
    metadata_path = cache_root / f"{key}.json"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"browser response cache is missing {url}")
    raw = raw_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_url") != url:
        raise ValueError(f"browser response cache URL mismatch: {url}")
    if int(metadata.get("http_status", 0)) != 200:
        raise ValueError(f"browser response cache is not HTTP 200: {url}")
    if metadata.get("raw_sha256") != _sha256_bytes(raw):
        raise ValueError(f"browser response cache hash mismatch: {url}")
    return raw, {
        "http_status": "200",
        "final_url": str(metadata.get("final_url", url)),
        "content_type": str(metadata.get("content_type", "text/html")),
        "last_modified": str(metadata.get("last_modified", "")),
        "etag": str(metadata.get("etag", "")),
        "retrieved_at": str(metadata["retrieved_at"]),
        "transport": str(metadata.get("transport", "browser_response_cache")),
    }


def _parse_filing_date(index_html: bytes, *, index_url: str) -> str:
    normalized = normalize_edgar_filing_html(index_html)
    match = re.search(r"\bFiling Date\s+(\d{4}-\d{2}-\d{2})\b", normalized, re.IGNORECASE)
    if match is None:
        match = re.search(r"\bFiling Date\s*\n\s*(\d{4}-\d{2}-\d{2})\b", normalized, re.IGNORECASE)
    if match is None:
        raise ValueError(f"could not parse filing date from SEC filing index: {index_url}")
    return match.group(1)


def _literal_supports(
    term: str,
    source_runtime: Sequence[tuple[dict[str, Any], str, str]],
) -> list[dict[str, Any]]:
    supports: list[dict[str, Any]] = []
    folded_term = term.casefold()
    for source, upload, _normalized in source_runtime:
        folded_upload = upload.casefold()
        start = folded_upload.find(folded_term)
        if start < 0:
            continue
        span = source["upload_span_in_normalized"]
        normalized_start = int(span["char_start"]) + start
        normalized_end = normalized_start + len(term)
        supports.append(
            {
                "source_id": source["source_id"],
                "normalized_char_start": normalized_start,
                "normalized_char_end": normalized_end,
                "match_strategy": "case_insensitive_literal",
            }
        )
    return supports


def _literal_support_for_named_source(
    claim: str,
    source_name: str,
    source_runtime: Sequence[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    matching_sources = [
        (source, upload, normalized)
        for source, upload, normalized in source_runtime
        if str(source["name"]) == source_name
    ]
    if len(matching_sources) != 1:
        raise ValueError(
            f"contradiction side source {source_name!r} must name exactly one case source"
        )
    supports = _literal_supports(claim, matching_sources)
    if len(supports) != 1:
        raise ValueError(
            f"contradiction claim lacks exact upload support in {source_name!r}: {claim!r}"
        )
    return supports[0]


def _case_annotations(
    case: dict[str, Any],
    source_runtime: Sequence[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    fact_annotations = []
    for term in case["planted_fact_terms"]:
        supports = _literal_supports(str(term), source_runtime)
        if not supports:
            raise ValueError(f"{case['id']}: planted fact term has no upload support: {term!r}")
        fact_annotations.append({"term": str(term), "support_spans": supports})

    answer_annotations = []
    answer_source_ids: set[str] = set()
    for term in case["answer_terms"]:
        supports = _literal_supports(str(term), source_runtime)
        if not supports:
            raise ValueError(f"{case['id']}: answer term has no upload support: {term!r}")
        answer_source_ids.update(str(support["source_id"]) for support in supports)
        answer_annotations.append({"term": str(term), "support_spans": supports})
    if len(answer_source_ids) != 1:
        raise ValueError(
            f"{case['id']}: answer target must resolve to exactly one upload source; "
            f"found {sorted(answer_source_ids)}"
        )

    gap_annotations = []
    for term in case["gap_terms"]:
        count = sum(normalized.casefold().count(str(term).casefold()) for _, _, normalized in source_runtime)
        gap_annotations.append(
            {
                "term": str(term),
                "full_packet_search_terms": [str(term)],
                "full_normalized_occurrence_count": count,
                "interpretation": (
                    "literal_absence" if count == 0 else "topic_present_but_decision_evidence_incomplete"
                ),
            }
        )

    contradiction_annotations = []
    for contradiction in case.get("contradictions", ()):
        sides = []
        for side in contradiction["sides"]:
            claim = str(side["claim"])
            source_name = str(side["source_name"])
            sides.append(
                {
                    "side": str(side["side"]),
                    "source_name": source_name,
                    "claim": claim,
                    "support_span": _literal_support_for_named_source(
                        claim,
                        source_name,
                        source_runtime,
                    ),
                }
            )
        contradiction_annotations.append(
            {
                "term": str(contradiction["term"]),
                "classification": str(contradiction["classification"]),
                "sides": sides,
            }
        )
    normalized_packet = "\n".join(normalized for _, _, normalized in source_runtime)
    exact_question_count = normalized_packet.casefold().count(
        str(case["unanswerable_question"]).casefold()
    )
    if exact_question_count:
        raise ValueError(
            f"{case['id']}: unanswerable evaluation question occurs literally in source text"
        )

    return {
        "facts": fact_annotations,
        "gaps": gap_annotations,
        "contradictions": contradiction_annotations,
        "answerable_question": {
            "question": case["answerable_question"],
            "answer_terms": answer_annotations,
            "unique_support_source_id": next(iter(answer_source_ids)),
            "support_scope": "exactly_one_case_upload_source",
        },
        "unanswerable_question": {
            "question": case["unanswerable_question"],
            "packet_search_scope": "all normalized full-text sources",
            "exact_question_occurrence_count": exact_question_count,
            "deterministic_status": "literal_question_absent",
            "subjective_unanswerability_status": "HUMAN_REQUIRED",
            "reason": (
                "The evaluation question is absent as a literal source statement. A human reviewer "
                "must confirm that the packet lacks enough evidence to answer it substantively."
            ),
        },
    }


def build_corpus(
    *,
    output_root: Path,
    user_agent: str,
    refresh: bool,
    extract_bytes: int,
    browser_cache: Path | None,
) -> dict[str, Any]:
    if not CASES:
        raise RuntimeError("the frozen EDGAR case plan is empty")
    if extract_bytes <= 0 or extract_bytes > MAX_SOURCE_BYTES:
        raise ValueError(f"--extract-bytes must be between 1 and {MAX_SOURCE_BYTES}")

    output_root.mkdir(parents=True, exist_ok=True)
    existing_manifest = _load_existing_manifest(output_root)
    client: SecClient | None = None

    def network_client() -> SecClient:
        nonlocal client
        if client is None:
            client = SecClient(user_agent=user_agent)
        return client

    manifest_cases: list[dict[str, Any]] = []
    filing_index_cache: dict[str, dict[str, Any]] = {}

    for case in CASES:
        case_id = str(case["id"])
        sources = list(case["sources"])
        if not 1 <= len(sources) <= 5:
            raise ValueError(f"{case_id}: expected one to five sources")
        manifest_sources: list[dict[str, Any]] = []
        source_runtime: list[tuple[dict[str, Any], str, str]] = []
        case_upload_bytes = 0
        for source_index, source in enumerate(sources, start=1):
            legacy_paths = _legacy_source_paths(output_root, case_id, source)
            previous = _existing_source_metadata(existing_manifest, case_id, source["name"])
            previous_raw_path = _previous_artifact_path(
                previous, key="raw_path", output_root=output_root
            )
            response_metadata: dict[str, str]
            cached_raw_path = previous_raw_path or (
                legacy_paths["raw"] if legacy_paths["raw"].is_file() else None
            )
            if cached_raw_path is not None and not refresh:
                raw = cached_raw_path.read_bytes()
                response_metadata = {
                    "http_status": str((previous or {}).get("http_status", 200)),
                    "final_url": str((previous or {}).get("final_url", source["url"])),
                    "content_type": str((previous or {}).get("content_type", "text/html")),
                    "last_modified": str((previous or {}).get("last_modified", "")),
                    "etag": str((previous or {}).get("etag", "")),
                    "retrieved_at": str((previous or {}).get("retrieved_at", "")),
                    "transport": str((previous or {}).get("retrieval_transport", "local_raw_cache")),
                }
            elif browser_cache is not None:
                raw, response_metadata = _read_response_cache(browser_cache, str(source["url"]))
            else:
                raw, response_metadata = network_client().get(str(source["url"]))
                response_metadata["retrieved_at"] = datetime.now(UTC).isoformat()
                response_metadata["transport"] = "python_urllib_https"
            if not response_metadata["retrieved_at"]:
                response_metadata["retrieved_at"] = datetime.now(UTC).isoformat()
            if response_metadata["http_status"] != "200":
                raise RuntimeError(f"{case_id}/{source['name']}: source is not HTTP 200")

            raw_sha256 = _sha256_bytes(raw)
            raw_path = _shared_artifact_path(
                output_root, kind="raw", sha256=raw_sha256, suffix=".html"
            )
            _write_bytes_if_changed(raw_path, raw)

            filing_index_url = str(source["filing_index_url"])
            accession = str(source["accession"])
            if accession not in filing_index_cache:
                previous_filing_date = str((previous or {}).get("filing_date", ""))
                if previous_filing_date and not refresh:
                    filing_index_cache[accession] = {
                        "filing_date": previous_filing_date,
                        "http_status": int((previous or {}).get("filing_index_http_status", 200)),
                    }
                else:
                    if browser_cache is not None:
                        index_body, index_response = _read_response_cache(
                            browser_cache, filing_index_url
                        )
                    else:
                        index_body, index_response = network_client().get(filing_index_url)
                    filing_index_cache[accession] = {
                        "filing_date": _parse_filing_date(index_body, index_url=filing_index_url),
                        "http_status": int(index_response["http_status"]),
                    }
            filing_index_metadata = filing_index_cache[accession]

            normalized = normalize_edgar_filing_html(raw)
            normalized_sha256 = _sha256_text(normalized)
            normalized_path = _shared_artifact_path(
                output_root,
                kind="normalized",
                sha256=normalized_sha256,
                suffix=".txt",
            )
            _write_text_if_changed(normalized_path, normalized)
            extract, char_start, char_end = select_verbatim_extract(
                normalized,
                anchors=tuple(source["anchors"]),
                max_bytes=extract_bytes,
            )
            upload_path = legacy_paths["upload"]
            _write_text_if_changed(upload_path, extract)
            upload_size = len(extract.encode("utf-8"))
            case_upload_bytes += upload_size
            manifest_source = {
                    "name": str(source["name"]),
                    "source_id": f"{case_id}-source-{source_index:02d}",
                    "case_id": case_id,
                    "role": str(source.get("role", "contract_chain_document")),
                    "legal_party": str(source["legal_party"]),
                    "cik": str(case["cik"]),
                    "accession_number": str(source["accession"]),
                    "form_type": str(source["form"]),
                    "filing_date": str(filing_index_metadata["filing_date"]),
                    "filing_period": str(source.get("filing_period", "")),
                    "exhibit_number": str(source.get("exhibit_number", "")),
                    "document_name": urllib.parse.urlparse(str(source["url"])).path.rsplit("/", 1)[-1],
                    "filing_index_url": filing_index_url,
                    "filing_index_http_status": int(filing_index_metadata["http_status"]),
                    "source_url": str(source["url"]),
                    "final_url": response_metadata["final_url"],
                    "retrieved_at": response_metadata["retrieved_at"],
                    "verified_at": response_metadata["retrieved_at"],
                    "retrieval_transport": response_metadata["transport"],
                    "http_status": int(response_metadata["http_status"]),
                    "content_type": response_metadata["content_type"],
                    "last_modified": response_metadata["last_modified"],
                    "etag": response_metadata["etag"],
                    "raw_path": _relative(raw_path, output_root),
                    "normalized_path": _relative(normalized_path, output_root),
                    "upload_path": _relative(upload_path, output_root),
                    "raw_sha256": raw_sha256,
                    "normalized_sha256": normalized_sha256,
                    "upload_sha256": _sha256_text(extract),
                    "raw_bytes": len(raw),
                    "normalized_bytes": len(normalized.encode("utf-8")),
                    "upload_bytes": upload_size,
                    "encoding": "utf-8",
                    "normalizer": "cornerstone_edgar_visible_text_v1",
                    "upload_span_in_normalized": {
                        "coordinate_system": "unicode_code_points",
                        "char_start": char_start,
                        "char_end": char_end,
                        "sha256": _sha256_text(extract),
                    },
                    "anchor_terms": list(source["anchors"]),
                    "sec_redaction_markers": sorted(
                        marker for marker in ("[***]", "[**]", "[*]") if marker in normalized
                    ),
                    "supersedes": list(source.get("supersedes", [])),
                    "incorporated_by_reference": list(source.get("incorporated_by_reference", [])),
                    "source_order": source_index,
                }
            manifest_sources.append(manifest_source)
            source_runtime.append((manifest_source, extract, normalized))
        if case_upload_bytes > MAX_CASE_BYTES:
            raise ValueError(f"{case_id}: upload bundle exceeds {MAX_CASE_BYTES} bytes")
        manifest_cases.append(
            {
                "id": case_id,
                "issuer": str(case["issuer"]),
                "cik": str(case["cik"]),
                "archetype": str(case["archetype"]),
                "as_of_date": max(source["filing_date"] for source in manifest_sources),
                "decision_question": str(case["decision_question"]),
                "decision_owner": str(case["decision_owner"]),
                "operational_decision": str(case["operational_decision"]),
                "source_count": len(manifest_sources),
                "upload_bundle_bytes": case_upload_bytes,
                "planted_fact_terms": list(case["planted_fact_terms"]),
                "gap_terms": list(case["gap_terms"]),
                "contradiction_terms": list(case.get("contradiction_terms", [])),
                "answerable_question": str(case["answerable_question"]),
                "answer_terms": list(case["answer_terms"]),
                "unanswerable_question": str(case["unanswerable_question"]),
                "document_relationships": list(case["document_relationships"]),
                "annotations": _case_annotations(case, source_runtime),
                "sources": manifest_sources,
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "corpus_id": CORPUS_ID,
        "language": "en",
        "target_cohort": "operational decision owners",
        "domain": "SEC EDGAR commercial contracts and issuer disclosures",
        "provenance_policy": (
            "Source payloads are verbatim substrings of deterministic full-text "
            "normalizations of official SEC filing HTML; no source text is paraphrased."
        ),
        "retrieval_policy": {
            "official_hosts": sorted(ALLOWED_SEC_HOSTS),
            "maximum_requests_per_second": MAX_REQUESTS_PER_SECOND,
            "identifying_user_agent_required": True,
            "user_agent_not_persisted": True,
        },
        "intake_limits": {
            "source_count_min": 1,
            "source_count_max": 5,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_case_bytes": MAX_CASE_BYTES,
        },
        "case_count": len(manifest_cases),
        "source_count": sum(case["source_count"] for case in manifest_cases),
        "human_review_case_ids": list(HUMAN_REVIEW_CASE_IDS),
        "cases": manifest_cases,
    }
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_bytes_if_changed(output_root / "manifest.json", encoded)
    freeze = {
        "schema_version": "cs.vs5_edgar_eval_freeze.v1",
        "corpus_id": CORPUS_ID,
        "manifest_path": "fixtures/vs5/edgar-eval/manifest.json",
        "manifest_sha256": _sha256_bytes(encoded),
        "case_count": len(manifest_cases),
        "source_count": manifest["source_count"],
        "language": "en",
        "change_policy": (
            "Any manifest, source URL, raw response, normalization, upload span, "
            "or hash change invalidates dependent Plane 2 evidence."
        ),
    }
    _write_text_if_changed(
        output_root / "freeze.json",
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n",
    )
    _remove_legacy_source_copies(output_root)
    _remove_unreferenced_shared_objects(output_root, manifest)
    validate_corpus(output_root=output_root)
    return manifest


def _require_file(output_root: Path, relative: str) -> Path:
    path = REPOSITORY_ROOT / relative
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(f"manifest path escapes corpus root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_corpus(*, output_root: Path) -> dict[str, int]:
    manifest_path = output_root / "manifest.json"
    freeze_path = output_root / "freeze.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected EDGAR manifest schema")
    if manifest.get("corpus_id") != CORPUS_ID or freeze.get("corpus_id") != CORPUS_ID:
        raise ValueError("unexpected EDGAR corpus id")
    if manifest.get("case_count") != len(manifest.get("cases", [])):
        raise ValueError("case_count does not match cases")
    if len(manifest["cases"]) != 25:
        raise ValueError(f"expected 25 cases, found {len(manifest['cases'])}")
    if manifest.get("human_review_case_ids") != list(HUMAN_REVIEW_CASE_IDS):
        raise ValueError("human-review cohort does not match the frozen first ten cases")
    if len(set(manifest["human_review_case_ids"])) != 10:
        raise ValueError("human-review cohort must contain ten unique cases")
    if freeze.get("manifest_sha256") != _sha256_bytes(manifest_bytes):
        raise ValueError("freeze manifest_sha256 does not match manifest bytes")
    if freeze.get("case_count") != 25:
        raise ValueError("freeze case_count does not match corpus contents")

    case_ids: set[str] = set()
    source_ids: set[str] = set()
    referenced_shared_paths: set[Path] = set()
    unique_raw_paths: set[Path] = set()
    unique_normalized_paths: set[Path] = set()
    source_count = 0
    contradiction_case_count = 0
    raw_bytes = normalized_bytes = upload_bytes = 0
    for case in manifest["cases"]:
        case_id = str(case["id"])
        if case_id in case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        if not case.get("gap_terms"):
            raise ValueError(f"{case_id}: gap_terms must not be empty")
        sources = case.get("sources", [])
        if not 1 <= len(sources) <= 5 or case.get("source_count") != len(sources):
            raise ValueError(f"{case_id}: invalid source count")
        case_bytes = 0
        source_by_id: dict[str, tuple[dict[str, Any], str, str]] = {}
        source_id_by_name: dict[str, str] = {}
        normalized_packet_parts: list[str] = []
        for source in sources:
            source_count += 1
            source_id = str(source["source_id"])
            if source_id in source_ids:
                raise ValueError(f"duplicate source id: {source_id}")
            source_ids.add(source_id)
            if source.get("case_id") != case_id:
                raise ValueError(f"{case_id}/{source.get('name')}: source case id mismatch")
            if source.get("http_status") != 200 or source.get("filing_index_http_status") != 200:
                raise ValueError(f"{case_id}/{source.get('name')}: SEC HTTP status is not 200")
            for url_key in ("source_url", "final_url", "filing_index_url"):
                parsed = urllib.parse.urlparse(str(source[url_key]))
                if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SEC_HOSTS:
                    raise ValueError(
                        f"{case_id}/{source.get('name')}: non-SEC URL in {url_key}"
                    )
            if not source.get("retrieved_at") or not source.get("verified_at"):
                raise ValueError(f"{case_id}/{source.get('name')}: missing retrieval timestamp")
            raw_path = _require_file(output_root, source["raw_path"])
            normalized_path = _require_file(output_root, source["normalized_path"])
            upload_path = _require_file(output_root, source["upload_path"])
            expected_raw_path = _shared_artifact_path(
                output_root,
                kind="raw",
                sha256=str(source["raw_sha256"]),
                suffix=".html",
            )
            expected_normalized_path = _shared_artifact_path(
                output_root,
                kind="normalized",
                sha256=str(source["normalized_sha256"]),
                suffix=".txt",
            )
            if raw_path.resolve() != expected_raw_path.resolve():
                raise ValueError(f"{case_id}/{source['name']}: raw path is not content-addressed")
            if normalized_path.resolve() != expected_normalized_path.resolve():
                raise ValueError(
                    f"{case_id}/{source['name']}: normalized path is not content-addressed"
                )
            if output_root / case_id / "upload" not in upload_path.parents:
                raise ValueError(f"{case_id}/{source['name']}: upload path is not case-specific")
            referenced_shared_paths.update((raw_path.resolve(), normalized_path.resolve()))
            unique_raw_paths.add(raw_path.resolve())
            unique_normalized_paths.add(normalized_path.resolve())
            raw = raw_path.read_bytes()
            normalized = normalized_path.read_text(encoding="utf-8")
            upload = upload_path.read_text(encoding="utf-8")
            if _sha256_bytes(raw) != source["raw_sha256"]:
                raise ValueError(f"{case_id}/{source['name']}: raw hash mismatch")
            if normalize_edgar_filing_html(raw) != normalized:
                raise ValueError(f"{case_id}/{source['name']}: raw normalization mismatch")
            if _sha256_text(normalized) != source["normalized_sha256"]:
                raise ValueError(f"{case_id}/{source['name']}: normalized hash mismatch")
            if _sha256_text(upload) != source["upload_sha256"]:
                raise ValueError(f"{case_id}/{source['name']}: upload hash mismatch")
            raw_size = len(raw)
            normalized_size = len(normalized.encode("utf-8"))
            upload_size = len(upload.encode("utf-8"))
            if (
                raw_size != source.get("raw_bytes")
                or normalized_size != source.get("normalized_bytes")
                or upload_size != source.get("upload_bytes")
            ):
                raise ValueError(f"{case_id}/{source['name']}: stored byte count mismatch")
            span = source["upload_span_in_normalized"]
            if span.get("coordinate_system") != "unicode_code_points":
                raise ValueError(f"{case_id}/{source['name']}: unsupported span coordinates")
            if normalized[span["char_start"]:span["char_end"]] != upload:
                raise ValueError(f"{case_id}/{source['name']}: upload span is not exact")
            if upload_size > MAX_SOURCE_BYTES:
                raise ValueError(f"{case_id}/{source['name']}: upload exceeds source limit")
            if span.get("sha256") != source["upload_sha256"]:
                raise ValueError(f"{case_id}/{source['name']}: upload span hash mismatch")
            legal_party = str(source.get("legal_party") or "")
            collapsed_party = re.sub(r"\s+", " ", legal_party).strip()
            collapsed_body = re.sub(r"\s+", " ", normalized)
            if not collapsed_party or collapsed_party not in collapsed_body:
                raise ValueError(
                    f"{case_id}/{source['name']}: legal_party is not an exact "
                    "whitespace-normalized preserved-body legal name"
                )
            for anchor in source["anchor_terms"]:
                if anchor.casefold() not in upload.casefold():
                    raise ValueError(f"{case_id}/{source['name']}: upload omitted anchor {anchor!r}")
            source_by_id[source_id] = (source, upload, normalized)
            source_name = str(source["name"])
            if source_name in source_id_by_name:
                raise ValueError(f"{case_id}: duplicate source name: {source_name}")
            source_id_by_name[source_name] = source_id
            normalized_packet_parts.append(normalized)
            case_bytes += upload_size
            raw_bytes += raw_size
            normalized_bytes += normalized_size
            upload_bytes += upload_size
        if case_bytes != case.get("upload_bundle_bytes") or case_bytes > MAX_CASE_BYTES:
            raise ValueError(f"{case_id}: upload bundle byte count mismatch or overflow")
        if case.get("as_of_date") != max(source["filing_date"] for source in sources):
            raise ValueError(f"{case_id}: as_of_date is not the latest filing date")

        annotations = case.get("annotations", {})
        for annotation in annotations.get("facts", []):
            term = str(annotation["term"])
            for support in annotation.get("support_spans", []):
                support_source_id = str(support["source_id"])
                if support_source_id not in source_by_id:
                    raise ValueError(f"{case_id}: annotation references unknown source")
                normalized = source_by_id[support_source_id][2]
                supported_text = normalized[
                    int(support["normalized_char_start"]):int(support["normalized_char_end"])
                ]
                if supported_text.casefold() != term.casefold():
                    raise ValueError(f"{case_id}: annotation span does not support {term!r}")

        contradiction_terms = list(case.get("contradiction_terms") or [])
        contradiction_annotations = list(annotations.get("contradictions") or [])
        if contradiction_terms:
            contradiction_case_count += 1
        if [item.get("term") for item in contradiction_annotations] != contradiction_terms:
            raise ValueError(f"{case_id}: contradiction terms and annotations differ")
        for contradiction in contradiction_annotations:
            if contradiction.get("classification") not in {
                "contradiction",
                "scope_difference",
                "supersession",
            }:
                raise ValueError(f"{case_id}: unsupported contradiction classification")
            sides = contradiction.get("sides")
            if not isinstance(sides, list) or len(sides) != 2:
                raise ValueError(f"{case_id}: contradiction must have exactly two sides")
            if [side.get("side") for side in sides] != ["prior", "current"]:
                raise ValueError(
                    f"{case_id}: contradiction sides must be ordered prior then current"
                )
            side_source_ids: set[str] = set()
            side_claims: set[str] = set()
            for side in sides:
                source_name = str(side.get("source_name") or "")
                expected_source_id = source_id_by_name.get(source_name)
                support = side.get("support_span")
                claim = str(side.get("claim") or "")
                if expected_source_id is None or not isinstance(support, dict) or not claim:
                    raise ValueError(f"{case_id}: incomplete contradiction side")
                support_source_id = str(support.get("source_id") or "")
                if support_source_id != expected_source_id:
                    raise ValueError(f"{case_id}: contradiction side source mismatch")
                source, upload, normalized = source_by_id[support_source_id]
                start = int(support.get("normalized_char_start", -1))
                end = int(support.get("normalized_char_end", -1))
                upload_span = source["upload_span_in_normalized"]
                if (
                    start < int(upload_span["char_start"])
                    or end > int(upload_span["char_end"])
                    or end <= start
                    or normalized[start:end] != claim
                ):
                    raise ValueError(
                        f"{case_id}: contradiction side lacks exact in-upload support"
                    )
                relative_start = start - int(upload_span["char_start"])
                relative_end = end - int(upload_span["char_start"])
                if upload[relative_start:relative_end] != claim:
                    raise ValueError(
                        f"{case_id}: contradiction side upload text does not match claim"
                    )
                side_source_ids.add(support_source_id)
                side_claims.add(claim.casefold())
            if len(side_source_ids) != 2 or len(side_claims) != 2:
                raise ValueError(
                    f"{case_id}: contradiction sides need distinct claims and sources"
                )

        answer_source_ids: set[str] = set()
        for term in case.get("answer_terms", []):
            supporting_ids = {
                source_id
                for source_id, (_source, upload, _normalized) in source_by_id.items()
                if str(term).casefold() in upload.casefold()
            }
            if not supporting_ids:
                raise ValueError(f"{case_id}: answer term lacks literal upload support: {term!r}")
            answer_source_ids.update(supporting_ids)
        if len(answer_source_ids) != 1:
            raise ValueError(
                f"{case_id}: answer target is not unique to one upload source: "
                f"{sorted(answer_source_ids)}"
            )
        answer_annotation = annotations.get("answerable_question", {})
        if answer_annotation.get("unique_support_source_id") != next(iter(answer_source_ids)):
            raise ValueError(f"{case_id}: answer support-source annotation mismatch")
        for annotated_term in answer_annotation.get("answer_terms", []):
            term = str(annotated_term["term"])
            for support in annotated_term.get("support_spans", []):
                source_id = str(support["source_id"])
                normalized = source_by_id[source_id][2]
                supported_text = normalized[
                    int(support["normalized_char_start"]):int(support["normalized_char_end"])
                ]
                if supported_text.casefold() != term.casefold():
                    raise ValueError(f"{case_id}: answer span does not support {term!r}")

        normalized_packet = "\n".join(normalized_packet_parts)
        if str(case["unanswerable_question"]).casefold() in normalized_packet.casefold():
            raise ValueError(f"{case_id}: unanswerable question occurs in normalized packet")
        unanswerable_annotation = annotations.get("unanswerable_question", {})
        if (
            unanswerable_annotation.get("exact_question_occurrence_count") != 0
            or unanswerable_annotation.get("subjective_unanswerability_status") != "HUMAN_REQUIRED"
        ):
            raise ValueError(f"{case_id}: invalid unanswerable-question annotation")

    if not set(HUMAN_REVIEW_CASE_IDS).issubset(case_ids):
        raise ValueError("human-review cohort references an unknown case")
    if contradiction_case_count < 3:
        raise ValueError("corpus requires at least three two-sided contradiction cases")
    if source_count != manifest.get("source_count") or source_count != freeze.get("source_count"):
        raise ValueError("source_count does not match corpus contents")
    physical_shared_files = {
        path.resolve()
        for path in (output_root / "_shared").rglob("*")
        if path.is_file()
    }
    if physical_shared_files != referenced_shared_paths:
        raise ValueError("shared content-addressed store has missing or unreferenced objects")
    if len(unique_raw_paths) != 72 or len(unique_normalized_paths) != 72:
        raise ValueError("frozen corpus must preserve exactly 72 distinct SEC documents")
    for case_id in case_ids:
        for legacy_name in ("raw", "normalized"):
            if (output_root / case_id / legacy_name).exists():
                raise ValueError(f"{case_id}: legacy case-local {legacy_name} directory remains")
    return {
        "case_count": len(case_ids),
        "source_count": source_count,
        "legal_party_binding_count": source_count,
        "two_sided_contradiction_case_count": contradiction_case_count,
        "unique_raw_file_count": len(unique_raw_paths),
        "unique_normalized_file_count": len(unique_normalized_paths),
        "raw_bytes": raw_bytes,
        "normalized_bytes": normalized_bytes,
        "upload_bytes": upload_bytes,
        "physical_raw_bytes": sum(path.stat().st_size for path in unique_raw_paths),
        "physical_normalized_bytes": sum(
            path.stat().st_size for path in unique_normalized_paths
        ),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    parser.add_argument("--refresh", action="store_true", help="refetch even when raw files exist")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--extract-bytes", type=int, default=DEFAULT_EXTRACT_BYTES)
    parser.add_argument(
        "--browser-cache",
        type=Path,
        help="import HTTP-200 response bodies from a fail-closed temporary browser cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_root = args.output_root.resolve()
    if args.validate_only:
        result = validate_corpus(output_root=output_root)
    else:
        manifest = build_corpus(
            output_root=output_root,
            user_agent=args.user_agent,
            refresh=args.refresh,
            extract_bytes=args.extract_bytes,
            browser_cache=args.browser_cache.resolve() if args.browser_cache else None,
        )
        result = validate_corpus(output_root=output_root)
        result["manifest_case_count"] = manifest["case_count"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
