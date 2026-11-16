-- Selectivity workload over the ClickBench hits table.
-- Query labels are stable identifiers, not claims about current row counts or pruning.
-- Measure selectivity, dictionary cardinality, and pruning on the supplied input.

-- pr-1rg EventDate = 2013-07-23
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "EventDate" = TIMESTAMP '2013-07-23 00:00:00';

-- pr-6rg EventDate = 2013-07-02
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "EventDate" = TIMESTAMP '2013-07-02 00:00:00';

-- pr-18rg EventDate = 2013-07-15
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "EventDate" = TIMESTAMP '2013-07-15 00:00:00';

-- pr-53rg EventDate = 2013-07-03
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "EventDate" = TIMESTAMP '2013-07-03 00:00:00';

-- ndl-10 CounterID = 52
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "CounterID" = 52;

-- ndl-1k CounterID = 2032
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "CounterID" = 2032;

-- ndl-9k CounterID = 1490
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "CounterID" = 1490;

-- ndl-114k CounterID = 1830
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "CounterID" = 1830;

-- ndl-5m CounterID = 3922
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "CounterID" = 3922;

-- ct-107k RegionID = 59
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "RegionID" = 59;

-- ct-9k RegionID = 10483
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits WHERE "RegionID" = 10483;

-- scan-all No filter
SELECT count(*), sum(length("URL")), sum(length("Title")), sum(length("Referer")) FROM hits;
