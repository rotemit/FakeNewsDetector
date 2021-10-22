import React from 'react'
import './App.css';

import {HomePage} from './Pages/HomePage'
import {EnterProfile} from './Pages/EnterProfile'
import {ScanPost} from './Pages/ScanPost'
import { Route, Switch, withRouter} from "react-router-dom";

function App() {
    return (
      <Switch>
          <Route exact path='/' component={HomePage} />
          <Route path='/EnterProfile' component={EnterProfile} />
          <Route path='/ScanPost' component={ScanPost} />
         
      </Switch>
    );
}


export default withRouter(App);
