var placeholder = document.getElementById("trades_placeholder")
var pusher = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('live_trades');
var i = 0;

trades_channel.bind('trade', function(data) {
    if (i == 0) {
        placeholder.innerHTML = '';
    }
    child = document.createElement("div");
    child.innerHTML = data['id'] + ': ' + data['amount'] + ' BTC @ ' + data['price'] + ' USD';
    placeholder.appendChild(child);
    i++;
});
