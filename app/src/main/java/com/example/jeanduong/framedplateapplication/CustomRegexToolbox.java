package com.example.jeanduong.framedplateapplication;

public class CustomRegexToolbox
{
    private String symbole_anchor = "sy\\s*[:\\.]?";
    private String symbole_base = "[0-9]+";

    //private String serial_anchor = "s[eé]r(ie)?\\s*[:\\.]?";
    private String serial_anchor = "s[eé]r(ie)?\\s*[:\\.]?";
    private String serial_base = "[A-Z]+";

    private String manufacturer_anchor = "n[o°]?\\s*[:\\.]?";
    private String manufacturer_base = "([0-9]\\s*){5}";

    private String warranty_anchor = "gar(antie)?\\s*[:\\.]?";
    private String warranty_base = "[0-9]{1,2}(\\s*ans?)?";

    private String commission_anchor = "date";
    private String commission_base = "(1[012]|0[1-9]|[1-9])\\s*[/\\-]\\s*([0-9]{2})?[0-9]{2}";

    public CustomRegexToolbox(){super();};

    public String getSymbole_anchor() {
        return symbole_anchor;
    }

    public String getSymbole_base() {
        return symbole_base;
    }

    public String getSerial_anchor() {
        return serial_anchor;
    }

    public String getSerial_base() {
        return serial_base;
    }

    public String getManufacturer_anchor() {
        return manufacturer_anchor;
    }

    public String getManufacturer_base() {
        return manufacturer_base;
    }

    public String getWarranty_anchor() {
        return warranty_anchor;
    }

    public String getWarranty_base() {
        return warranty_base;
    }

    public String getCommission_anchor() {
        return commission_anchor;
    }

    public String getCommission_base() {
        return commission_base;
    }
}
