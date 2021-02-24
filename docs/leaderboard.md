---
layout: sfda
---
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/dataTables.bootstrap4.min.css">
<script src="https://code.jquery.com/jquery-3.4.1.slim.min.js"
  integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n"
  crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
  integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
  crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"
  integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6"
  crossorigin="anonymous"></script>
<script type="text/javascript" charset="utf8" src="https://apis.google.com/js/api.js"></script>
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/dataTables.bootstrap4.min.js"></script>
<script type="text/javascript" language="javascript" src="assests/js/leaderboard.js"></script>

<div style="width: 50%; padding: 20px; float: left;">
<h3>Negation Detection</h3>    
<table id="negation" class="table table-striped table-bordered">
  <thead>	    
    <tr>
      <th>System</th><th>Precision</th><th>Recall</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MedAI-1</td><td>0.902</td><td>0.756</td><td>0.822</td></tr>
    <tr><td>MedAI-2</td><td>0.902</td><td>0.756</td><td>0.822</td></tr>
    <tr><td>UArizona-1</td><td>0.88</td><td>0.68</td><td>0.767</td></tr>
    <tr><td>BLCUFIGHT-2</td><td>0.913</td><td>0.616</td><td>0.736</td></tr>
    <tr><td>IITK-2</td><td>0.876</td><td>0.624</td><td>0.729</td></tr>
    <tr><td>Boom-1</td><td>0.929</td><td>0.597</td><td>0.727</td></tr>
    <tr><td>IITK-1</td><td>0.939</td><td>0.566</td><td>0.706</td></tr>
    <tr><td>BLCUFIGHT-1</td><td>0.528</td><td>0.639</td><td>0.578</td></tr>
    <tr><td>Baseline-2</td><td>0.908</td><td>0.611</td><td>0.73</td></tr>
    <tr><td>Baseline-1</td><td>0.917</td><td>0.516</td><td>0.66</td></tr>
  </tbody>
</table>
</div>

<div style="width: 50%; padding: 20px; float: left;">
<h3>Time Expression Recognition</h3>    
<table id="timex" class="table table-striped table-bordered">
  <thead>	    
    <tr>
      <th>System</th><th>Precision</th><th>Recall</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>BLCUFIGHT-1</td><td>0.847</td><td>0.785</td><td>0.815</td></tr>
    <tr><td>Self-Adapter-1</td><td>0.873</td><td>0.757</td><td>0.811</td></tr>
    <tr><td>BLCUFIGHT-2</td><td>0.834</td><td>0.787</td><td>0.81</td></tr>
    <tr><td>YNU-HPCC-2</td><td>0.817</td><td>0.791</td><td>0.803</td></tr>
    <tr><td>Self-Adapter-2</td><td>0.839</td><td>0.76</td><td>0.797</td></tr>
    <tr><td>PTST-UoM-1</td><td>0.901</td><td>0.713</td><td>0.796</td></tr>
    <tr><td>UArizona-1</td><td>0.786</td><td>0.804</td><td>0.795</td></tr>
    <tr><td>UArizona-2</td><td>0.783</td><td>0.807</td><td>0.795</td></tr>
    <tr><td>Boom-1</td><td>0.869</td><td>0.732</td><td>0.795</td></tr>
    <tr><td>KISNLP-1</td><td>0.81</td><td>0.777</td><td>0.793</td></tr>
    <tr><td>KISNLP-2</td><td>0.798</td><td>0.764</td><td>0.781</td></tr>
    <tr><td>YNU-HPCC-1</td><td>0.872</td><td>0.655</td><td>0.748</td></tr>
    <tr><td>Baseline-2</td><td>0.827</td><td>0.782</td><td>0.804</td></tr>
    <tr><td>Baseline-1</td><td>0.849</td><td>0.746</td><td>0.794</td></tr>
  </tbody>
</table>
</div>
