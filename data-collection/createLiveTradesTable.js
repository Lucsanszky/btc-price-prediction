/*
** Source: http://docs.aws.amazon.com/amazondynamodb/latest/gettingstartedguide/GettingStarted.NodeJs.01.html 
*/

var AWS = require("aws-sdk");

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var dynamodb = new AWS.DynamoDB();

var params = {
    TableName : "TradesLive",
    KeySchema: [       
        { AttributeName: "TradeID", KeyType: "HASH"},
        { AttributeName: "Date", KeyType: "RANGE" }  
    ],
    AttributeDefinitions: [       
        { AttributeName: "TradeID", AttributeType: "N" },
        { AttributeName: "Date", AttributeType: "S" }
    ],
    ProvisionedThroughput: {       
        ReadCapacityUnits: 5, 
        WriteCapacityUnits: 5
    }
};

dynamodb.createTable(params, function(err, data) {
    if (err) {
        console.error("Unable to create table. Error JSON:", JSON.stringify(err, null, 2));
    } else {
        console.log("Created table. Table description JSON:", JSON.stringify(data, null, 2));
    }
});