var express = require('express');
var router = express.Router();

let invokeRest = require('./rest/invoke-cc');

router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

router.stack = router.stack.concat(invokeRest.stack);

module.exports = router;
