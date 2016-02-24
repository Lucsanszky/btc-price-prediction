var AWS = require("aws-sdk");

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var docClient = new AWS.DynamoDB.DocumentClient();

var params = {
    TableName: "OrderBook"
};

docClient.scan(params, function(err, data) {
    if (err)
        console.log(JSON.stringify(err, null, 2));
    else
        console.log(JSON.stringify(data, null, 2));
});