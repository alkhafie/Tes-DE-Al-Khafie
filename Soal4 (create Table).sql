CREATE TABLE ecommerce_sessions (
    channelGrouping VARCHAR(255),
    country VARCHAR(255),
    fullVisitorId VARCHAR(255),
    timeOnSite INT,
    pageviews INT,
    sessionQualityDim INT,
    v2ProductName VARCHAR(255),
    productRevenue NUMERIC,
    productQuantity INT,
    productRefundAmount NUMERIC,
    PRIMARY KEY (fullVisitorId)
);

