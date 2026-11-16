-- Strings workload over the ClickBench hits table.
-- Query labels are stable identifiers, not claims about current row counts or pruning.
-- Measure selectivity, dictionary cardinality, and pruning on the supplied input.

-- lk-646 URL substring predicate: google
SELECT count(*) FROM hits WHERE "URL" LIKE '%google%';

-- lk-4k URL substring predicate: irkutsk
SELECT count(*) FROM hits WHERE "URL" LIKE '%irkutsk%';

-- lk-5m URL substring predicate: .ru/
SELECT count(*) FROM hits WHERE "URL" LIKE '%.ru/%';

-- lk-title Title substring predicate: -
SELECT count(*) FROM hits WHERE "Title" LIKE '%-%';

-- lk-proj URL predicate with projected strings and a deterministic LIMIT
SELECT "URL", "Title" FROM hits WHERE "URL" LIKE '%irkutsk%' ORDER BY "URL", "Title" LIMIT 20;

-- dict-eq BrowserLanguage equality
SELECT count(*) FROM hits WHERE "BrowserLanguage" = 'S0';

-- dict-eq2 BrowserCountry equality
SELECT count(*) FROM hits WHERE "BrowserCountry" = 'h1';

-- dict-neq BrowserLanguage inequality
SELECT count(*) FROM hits WHERE "BrowserLanguage" <> 'E3';

-- dict-gb Group by BrowserLanguage
SELECT "BrowserLanguage", count(*) FROM hits GROUP BY "BrowserLanguage" ORDER BY 2 DESC, 1 LIMIT 20;

-- dict-gb2 Group by PageCharset
SELECT "PageCharset", count(*) FROM hits GROUP BY "PageCharset" ORDER BY 2 DESC, 1 LIMIT 20;

-- hi-gb Group by URL
SELECT "URL", count(*) FROM hits GROUP BY "URL" ORDER BY 2 DESC, 1 LIMIT 20;

-- sp-ne Nonempty SearchPhrase
SELECT count(*) FROM hits WHERE "SearchPhrase" <> '';

-- str-len Total string length without a filter
SELECT sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits;
