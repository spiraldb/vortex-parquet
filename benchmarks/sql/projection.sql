-- Projection width sweep over the ClickBench `hits` table.
--
-- Every query is a full scan with no filter, so the only variable is HOW MANY COLUMNS are
-- decoded. `sum()` is used rather than `max()` on purpose: DataFusion's AggregateStatistics rule
-- can answer MIN/MAX from Parquet row-group statistics without decoding the column at all, which
-- would make a projection-width benchmark measure nothing.
--
-- Numeric and string families are kept separate at each width because their decode costs differ by
-- more than the widths do -- a string column carries a dictionary and a varying-length payload.

-- w1-num 1 numeric column(s), full scan
SELECT sum("WatchID") FROM hits;

-- w5-num 5 numeric column(s), full scan
SELECT sum("WatchID"), sum("JavaEnable"), sum("GoodEvent"), sum("CounterID"), sum("ClientIP") FROM hits;

-- w20-num 20 numeric column(s), full scan
SELECT sum("WatchID"), sum("JavaEnable"), sum("GoodEvent"), sum("CounterID"), sum("ClientIP"), sum("RegionID"), sum("UserID"), sum("CounterClass"), sum("OS"), sum("UserAgent"), sum("IsRefresh"), sum("RefererCategoryID"), sum("RefererRegionID"), sum("URLCategoryID"), sum("URLRegionID"), sum("ResolutionWidth"), sum("ResolutionHeight"), sum("ResolutionDepth"), sum("FlashMajor"), sum("FlashMinor") FROM hits;

-- w1-str 1 string column(s), full scan
SELECT sum(length("Title")) FROM hits;

-- w5-str 5 string column(s), full scan
SELECT sum(length("Title")), sum(length("URL")), sum(length("Referer")), sum(length("FlashMinor2")), sum(length("UserAgentMinor")) FROM hits;

-- w20-str 20 string column(s), full scan
SELECT sum(length("Title")), sum(length("URL")), sum(length("Referer")), sum(length("FlashMinor2")), sum(length("UserAgentMinor")), sum(length("MobilePhoneModel")), sum(length("Params")), sum(length("SearchPhrase")), sum(length("PageCharset")), sum(length("OriginalURL")), sum(length("HitColor")), sum(length("BrowserLanguage")), sum(length("BrowserCountry")), sum(length("SocialNetwork")), sum(length("SocialAction")), sum(length("SocialSourcePage")), sum(length("ParamOrderID")), sum(length("ParamCurrency")), sum(length("OpenstatServiceName")), sum(length("OpenstatCampaignID")) FROM hits;

-- w105-all every one of the 105 leaf columns, full scan
SELECT sum("WatchID"), sum("JavaEnable"), sum(length("Title")), sum("GoodEvent"), sum(extract(epoch from "EventTime")), sum(extract(epoch from "EventDate")), sum("CounterID"), sum("ClientIP"), sum("RegionID"), sum("UserID"), sum("CounterClass"), sum("OS"), sum("UserAgent"), sum(length("URL")), sum(length("Referer")), sum("IsRefresh"), sum("RefererCategoryID"), sum("RefererRegionID"), sum("URLCategoryID"), sum("URLRegionID"), sum("ResolutionWidth"), sum("ResolutionHeight"), sum("ResolutionDepth"), sum("FlashMajor"), sum("FlashMinor"), sum(length("FlashMinor2")), sum("NetMajor"), sum("NetMinor"), sum("UserAgentMajor"), sum(length("UserAgentMinor")), sum("CookieEnable"), sum("JavascriptEnable"), sum("IsMobile"), sum("MobilePhone"), sum(length("MobilePhoneModel")), sum(length("Params")), sum("IPNetworkID"), sum("TraficSourceID"), sum("SearchEngineID"), sum(length("SearchPhrase")), sum("AdvEngineID"), sum("IsArtifical"), sum("WindowClientWidth"), sum("WindowClientHeight"), sum("ClientTimeZone"), sum(extract(epoch from "ClientEventTime")), sum("SilverlightVersion1"), sum("SilverlightVersion2"), sum("SilverlightVersion3"), sum("SilverlightVersion4"), sum(length("PageCharset")), sum("CodeVersion"), sum("IsLink"), sum("IsDownload"), sum("IsNotBounce"), sum("FUniqID"), sum(length("OriginalURL")), sum("HID"), sum("IsOldCounter"), sum("IsEvent"), sum("IsParameter"), sum("DontCountHits"), sum("WithHash"), sum(length("HitColor")), sum(extract(epoch from "LocalEventTime")), sum("Age"), sum("Sex"), sum("Income"), sum("Interests"), sum("Robotness"), sum("RemoteIP"), sum("WindowName"), sum("OpenerName"), sum("HistoryLength"), sum(length("BrowserLanguage")), sum(length("BrowserCountry")), sum(length("SocialNetwork")), sum(length("SocialAction")), sum("HTTPError"), sum("SendTiming"), sum("DNSTiming"), sum("ConnectTiming"), sum("ResponseStartTiming"), sum("ResponseEndTiming"), sum("FetchTiming"), sum("SocialSourceNetworkID"), sum(length("SocialSourcePage")), sum("ParamPrice"), sum(length("ParamOrderID")), sum(length("ParamCurrency")), sum("ParamCurrencyID"), sum(length("OpenstatServiceName")), sum(length("OpenstatCampaignID")), sum(length("OpenstatAdID")), sum(length("OpenstatSourceID")), sum(length("UTMSource")), sum(length("UTMMedium")), sum(length("UTMCampaign")), sum(length("UTMContent")), sum(length("UTMTerm")), sum(length("FromTag")), sum("HasGCLID"), sum("RefererHash"), sum("URLHash"), sum("CLID") FROM hits;

-- nar-sel Narrow projection behind a CounterID filter
SELECT sum("ResolutionWidth") FROM hits WHERE "CounterID" = 1490;
