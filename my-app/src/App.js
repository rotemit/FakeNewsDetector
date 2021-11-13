import './App.css';
import {ScanPost} from './Pages/ScanPost'
import {Login} from './Pages/Login'
import { About } from './Pages/About';
import React, { useState, useEffect } from "react";
import background from "./image.jpeg";
import image from "./istockphoto-1176082691-612x612.jpg"

const Link = ({ className, href, children }) => {
  const onClick = (event) => {
    if (event.metaKey || event.ctrlKey) {
      return;
    }

    event.preventDefault();
    window.history.pushState({}, '', href);

    const navEvent = new PopStateEvent('popstate');
    window.dispatchEvent(navEvent);
  };

  return (
    <a onClick={onClick} className={className} href={href}>
      {children}
    </a>
  );
};


const Header = () => {
  return (
    <div className="ui menu">
      <Link href="/" className="item">
        Login
      </Link>
      <Link href="/ScanPost" className="item">
        Scan Post
      </Link>
      <Link href="/About" className="item">
        About
      </Link>
    </div>
  );
};

const Route = ({ path, children }) => {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    const onLocationChange = () => {
      setCurrentPath(window.location.pathname);
    };

    window.addEventListener('popstate', onLocationChange);

    return () => {
      window.removeEventListener('popstate', onLocationChange);
    };
  }, []);

  return currentPath === path ? children : null;
};

const App = () => {
  return (
    <div style={{ backgroundImage: `url(${background})` }}>
      <Header />
      <Route path="/About">
        <About />
      </Route>
      <Route path="/ScanPost">
        <ScanPost  />
      </Route>
      <Route path="/">
        <Login  />
      </Route>
  </div>
  )
}


// export default withRouter(App);
export default App;
