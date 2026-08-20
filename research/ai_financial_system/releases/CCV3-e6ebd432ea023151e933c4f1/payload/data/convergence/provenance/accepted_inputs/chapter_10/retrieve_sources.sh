#!/usr/bin/env bash
set -euo pipefail

lane_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="${lane_dir}/source_archive"
metadata_dir="${lane_dir}/response_metadata"
user_agent="TradeBot-AIFinancialSystemResearch/1.0 (personal research; lawful public-primary-source preservation)"

mkdir -p "${archive_dir}" "${metadata_dir}"

retrieve() {
  local source_id="$1"
  local extension="$2"
  local url="$3"
  local blob="${archive_dir}/${source_id}_v0001.${extension}"
  local headers="${metadata_dir}/${source_id}_v0001.headers"
  local transfer="${metadata_dir}/${source_id}_v0001.curl.json"
  local failure="${metadata_dir}/${source_id}_v0001.failure.txt"

  if [[ -f "${failure}" ]] || (
    [[ -f "${blob}" && -f "${headers}" && -f "${transfer}" ]]
  ); then
    return 0
  fi

  if curl \
    --location \
    --fail \
    --silent \
    --show-error \
    --remove-on-error \
    --retry 2 \
    --retry-delay 1 \
    --max-time 90 \
    --connect-timeout 15 \
    --user-agent "${user_agent}" \
    --dump-header "${headers}" \
    --output "${blob}" \
    --write-out "%output{${transfer}}%{json}" \
    "${url}"; then
    if [[ "${extension}" == "pdf" ]] && (
      [[ ! -s "${blob}" ]] || [[ "$(head -c 4 "${blob}")" != "%PDF" ]]
    ); then
      {
        printf '%s\n' \
          "source_id=${source_id}" \
          "attempt_id=${source_id}_v0001" \
          "original_url=${url}" \
          "status=invalid_pdf_response_hold" \
          "boundary=response_bytes_retained_but_not_treated_as_governing_document"
      } >"${failure}"
    fi
    return 0
  fi

  {
    printf '%s\n' \
      "source_id=${source_id}" \
      "attempt_id=${source_id}_v0001" \
      "original_url=${url}" \
      "status=retrieval_failed_after_bounded_retry" \
      "retry_count=2" \
      "max_time_seconds=90" \
      "boundary=access_control_respected_no_circumvention"
  } >"${failure}"
}

retrieve_version() {
  local source_id="$1"
  local version="$2"
  local extension="$3"
  local url="$4"
  local blob="${archive_dir}/${source_id}_${version}.${extension}"
  local headers="${metadata_dir}/${source_id}_${version}.headers"
  local transfer="${metadata_dir}/${source_id}_${version}.curl.json"
  local failure="${metadata_dir}/${source_id}_${version}.failure.txt"

  if [[ -f "${failure}" ]] || (
    [[ -f "${blob}" && -f "${headers}" && -f "${transfer}" ]]
  ); then
    return 0
  fi

  if curl \
    --location \
    --fail \
    --silent \
    --show-error \
    --remove-on-error \
    --retry 2 \
    --retry-delay 1 \
    --max-time 90 \
    --connect-timeout 15 \
    --user-agent "${user_agent}" \
    --dump-header "${headers}" \
    --output "${blob}" \
    --write-out "%output{${transfer}}%{json}" \
    "${url}"; then
    if [[ "${extension}" == "pdf" ]] && (
      [[ ! -s "${blob}" ]] || [[ "$(head -c 4 "${blob}")" != "%PDF" ]]
    ); then
      {
        printf '%s\n' \
          "source_id=${source_id}" \
          "attempt_id=${source_id}_${version}" \
          "original_url=${url}" \
          "status=invalid_pdf_response_hold" \
          "boundary=response_bytes_retained_but_not_treated_as_governing_document"
      } >"${failure}"
    fi
    return 0
  fi

  {
    printf '%s\n' \
      "source_id=${source_id}" \
      "attempt_id=${source_id}_${version}" \
      "original_url=${url}" \
      "status=retrieval_failed_after_bounded_retry" \
      "retry_count=2" \
      "max_time_seconds=90" \
      "boundary=access_control_respected_no_circumvention"
  } >"${failure}"
}

retrieve "SRC-0075" "html" "https://www.canada.ca/en/innovation-science-economic-development/news/2025/03/government-of-canada-finalizes-investment-to-support-canadian-born-ai-leader-cohere.html"
retrieve "SRC-0076" "html" "https://ised-isde.canada.ca/site/strategic-environmental-assessment/en/strategic-response-fund-cohere-compute-expansion-project-public-statement"
retrieve "SRC-0087" "pdf" "https://lpscpubvalence.lpsc.louisiana.gov/portal/PSC/ViewFile?fileId=AUl4qI5mSHA%3D"
retrieve "SRC-0097" "pdf" "https://mi-psc.my.site.com/sfc/servlet.shepherd/version/download/068cs00001MH71XAAT"
retrieve "SRC-0098" "pdf" "https://mi-psc.my.site.com/sfc/servlet.shepherd/version/download/068cs00001XbIDsAAN"
retrieve "SRC-0103" "pdf" "https://www.mlgw.com/images/content/files/pdf/new/xAI%202025%20Update.pdf"
retrieve "SRC-0136" "html" "https://www.commerce.gov/news/press-releases/2024/11/biden-harris-administration-announces-chips-incentives-award-tsmc"
retrieve "SRC-0137" "html" "https://www.commerce.gov/news/press-releases/2024/11/biden-harris-administration-announces-chips-incentives-award-intel"
retrieve "SRC-0138" "html" "https://www.commerce.gov/news/press-releases/2024/12/department-commerce-awards-chips-incentives-micron-idaho-and-new-york"
retrieve "SRC-0139" "html" "https://www.commerce.gov/news/press-releases/2024/12/biden-harris-administration-announces-chips-incentives-award-sk-hynix"
retrieve "SRC-0140" "html" "https://www.commerce.gov/news/press-releases/2024/12/biden-harris-administration-announces-chips-incentives-award-samsung"
retrieve "SRC-0141" "html" "https://www.sec.gov/Archives/edgar/data/50863/000005086325000129/intc-20250822.htm"
retrieve "SRC-0142" "html" "https://www.sec.gov/Archives/edgar/data/50863/000005086325000129/a08222025formex101stockand.htm"
retrieve "SRC-0143" "html" "https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm"
retrieve "SRC-0144" "pdf" "https://www.sec.gov/Archives/edgar/data/50863/000005086326000066/intc_courtesy-pdfa.pdf"
retrieve "SRC-0145" "pdf" "https://www.commerce.gov/sites/default/files/2026-02/DOCFY2025AFR_508.pdf"
retrieve "SRC-0146" "html" "https://www.sec.gov/Archives/edgar/data/723125/000072312526000015/mu-20260528.htm"
retrieve "SRC-0147" "html" "https://www.sec.gov/Archives/edgar/data/1046179/000162828026025362/tsm-20251231.htm"
retrieve "SRC-0148" "pdf" "https://www.congress.gov/119/plaws/publ21/PLAW-119publ21.pdf"
retrieve "SRC-0149" "html" "https://www.irs.gov/instructions/i3468"
retrieve "SRC-0150" "pdf" "https://www.irs.gov/pub/irs-drop/a-24-40.pdf"
retrieve "SRC-0151" "html" "https://www.pif.gov.sa/en/news-and-insights/press-releases/2025/hrh-crown-prince-launches-humain-as-global-ai-powerhouse/"
retrieve "SRC-0152" "html" "https://www.mubadala.com/en/news/abu-dhabi-launches-comprehensive-global-investment-strategy-on-artificial-intelligence"
retrieve "SRC-0153" "html" "https://www.mgx.ae/news-insights/aip-mgx-and-blackrocks-gip-close-acquisition-aligned-data-centers"
retrieve "SRC-0154" "html" "https://news.microsoft.com/source/2024/09/17/blackrock-global-infrastructure-partners-microsoft-and-mgx-launch-new-ai-partnership-to-invest-in-data-centers-and-supporting-power-infrastructure/"
retrieve "SRC-0155" "html" "https://digital-strategy.ec.europa.eu/en/news/eu-launches-investai-initiative-mobilise-eu200-billion-investment-artificial-intelligence"
retrieve "SRC-0156" "html" "https://digital-strategy.ec.europa.eu/en/library/memorandum-understanding-ai-gigafactories"

# Lawful official alternates are append-only and never erase the cited-URL failures.
retrieve_version "SRC-0136" "v0002" "html" "https://www.nist.gov/news-events/news/2024/11/biden-harris-administration-announces-chips-incentives-award-tsmc-arizona"
retrieve_version "SRC-0137" "v0002" "html" "https://www.nist.gov/news-events/news/2024/11/biden-harris-administration-announces-chips-incentives-award-intel-advance"
retrieve_version "SRC-0138" "v0002" "html" "https://www.nist.gov/news-events/news/2024/12/department-commerce-awards-chips-incentives-micron-idaho-and-new-york"
retrieve_version "SRC-0139" "v0002" "html" "https://www.nist.gov/news-events/news/2024/12/biden-harris-administration-announces-chips-incentives-award-sk-hynix"
retrieve_version "SRC-0140" "v0002" "html" "https://www.nist.gov/news-events/news/2024/12/biden-harris-administration-announces-chips-incentives-award-samsung"
retrieve_version "SRC-0145" "v0002" "pdf" "https://www.commerce.gov/sites/default/files/2026-02/DOC%20FY%202025%20AFR.pdf"
retrieve_version "SRC-0151" "v0002" "html" "https://www.pif.gov.sa/ar/news-and-insights/press-releases/2025/hrh-crown-prince-launches-humain-as-global-ai-powerhouse/"
retrieve_version "SRC-0152" "v0002" "html" "https://www.mubadala.com/en/news/abu-dhabi-launches-comprehensive-global-investment-strategy-on-artificial-intelligence?os=io....."
retrieve_version "SRC-0153" "v0002" "html" "https://aligneddc.com/press-release/aip-mgx-and-blackrocks-gip-close-acquisition-of-aligned-data-centers/"
retrieve_version "SRC-0156" "v0002" "html" "https://www.eib.org/en/press/all/2025-491-eib-group-and-european-commission-join-forces-to-finance-ai-gigafactories"
